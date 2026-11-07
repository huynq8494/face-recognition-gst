// scrfd_rknn_patched.cpp
// Full patched implementation to include kps decoding and follow Python SCRFD postprocessing logic.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <array>
#include <stdint.h>
#include <unordered_map>

#include "scrfd_rknn.h"
#include "file_utils.h"
#include "image_utils.h"

static const int DEFAULT_STRIDES_3[] = {8, 16, 32};
static const int DEFAULT_STRIDES_5[] = {8, 16, 32, 64, 128};

struct KeyHW {
    int h;
    int w;
    int stride;
    bool operator==(KeyHW const& o) const {
        return h==o.h && w==o.w && stride==o.stride;
    }
};
struct KeyHWHash {
    std::size_t operator()(KeyHW const& k) const noexcept {
        return ((size_t)k.h * 1000003u) ^ ((size_t)k.w * 9176u) ^ (size_t)k.stride;
    }
};

static std::unordered_map<KeyHW, std::vector<std::array<float,2>>, KeyHWHash> center_cache;

static void generate_anchor_centers_cached(std::vector<std::array<float,2>>& centers, int feat_h, int feat_w, int stride, int num_anchors) {
    KeyHW key{feat_h, feat_w, stride};
    auto it = center_cache.find(key);
    if (it != center_cache.end()) {
        centers = it->second;
        return;
    }
    centers.clear();
    centers.reserve(feat_h * feat_w * num_anchors);
    for (int gy = 0; gy < feat_h; gy++) {
        for (int gx = 0; gx < feat_w; gx++) {
            for (int a = 0; a < num_anchors; a++) {
                centers.push_back({(float)(gx * stride), (float)(gy * stride)});
            }
        }
    }
    if (center_cache.size() < 100) {
        center_cache.emplace(key, centers);
    }
}

static float iou_rect(float x1, float y1, float x2, float y2, float xx1, float yy1, float xx2, float yy2) {
    float ix1 = std::max(x1, xx1);
    float iy1 = std::max(y1, yy1);
    float ix2 = std::min(x2, xx2);
    float iy2 = std::min(y2, yy2);
    float iw = std::max(0.0f, ix2 - ix1 + 1.0f);
    float ih = std::max(0.0f, iy2 - iy1 + 1.0f);
    float inter = iw * ih;
    float a = (x2 - x1 + 1.0f) * (y2 - y1 + 1.0f);
    float b = (xx2 - xx1 + 1.0f) * (yy2 - yy1 + 1.0f);
    return inter / (a + b - inter + 1e-6f);
}

struct FaceDet { float x1,y1,x2,y2,score; };

static std::vector<int> nms_keep(const std::vector<FaceDet>& dets, float nms_thresh) {
    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b){ return dets[a].score > dets[b].score; });
    std::vector<int> keep;
    std::vector<char> suppressed(dets.size(), 0);
    for (size_t _i=0; _i<order.size(); _i++){
        int i = order[_i];
        if (suppressed[i]) continue;
        keep.push_back(i);
        for (size_t _j=_i+1; _j<order.size(); _j++){
            int j = order[_j];
            if (suppressed[j]) continue;
            float o = iou_rect(dets[i].x1,dets[i].y1,dets[i].x2,dets[i].y2, dets[j].x1,dets[j].y1,dets[j].x2,dets[j].y2);
            if (o > nms_thresh) suppressed[j] = 1;
        }
    }
    return keep;
}

// distance2bbox: centers (N,2), bbox_preds (N,4) where preds are raw offsets; multiply by stride inside
static void distance2bbox(const std::vector<std::array<float,2>>& centers, const float* bbox_preds, int N, int stride, std::vector<std::array<float,4>>& out_bboxes) {
    out_bboxes.resize(N);
    for (int i = 0; i < N; i++) {
        float cx = centers[i][0];
        float cy = centers[i][1];
        float l = bbox_preds[i*4 + 0] * stride;
        float t = bbox_preds[i*4 + 1] * stride;
        float r = bbox_preds[i*4 + 2] * stride;
        float b = bbox_preds[i*4 + 3] * stride;
        out_bboxes[i][0] = cx - l;
        out_bboxes[i][1] = cy - t;
        out_bboxes[i][2] = cx + r;
        out_bboxes[i][3] = cy + b;
    }
}

// distance2kps: centers (N,2), kps_preds (N,10) where preds are offsets; multiply by stride inside
static void distance2kps(const std::vector<std::array<float,2>>& centers, const float* kps_preds, int N, int stride, std::vector<std::array<float,10>>& out_kps_flat) {
    out_kps_flat.resize(N);
    for (int i = 0; i < N; i++) {
        float cx = centers[i][0];
        float cy = centers[i][1];
        for (int p = 0; p < 5; p++) {
            float dx = kps_preds[i*10 + p*2 + 0] * stride;
            float dy = kps_preds[i*10 + p*2 + 1] * stride;
            out_kps_flat[i][p*2 + 0] = cx + dx;
            out_kps_flat[i][p*2 + 1] = cy + dy;
        }
    }
}

int init_scrfd_model(const char* model_path, rknn_app_context_t* app_ctx) {
    int ret;
    int model_len = 0;
    char *model = NULL;
    rknn_context ctx = 0;

    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        printf("load_model fail! %s\n", model_path);
        return -1;
    }
    ret = rknn_init(&ctx, model, model_len, 0, NULL);
    free(model);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    rknn_input_output_num io_num;
    ret = rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        printf("rknn_query fail! ret=%d\n", ret);
        rknn_destroy(ctx);
        return -1;
    }

    rknn_tensor_attr *input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    rknn_tensor_attr *output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    for (int i=0;i<io_num.n_input;i++){
        input_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &input_attrs[i], sizeof(rknn_tensor_attr));
    }
    for (int i=0;i<io_num.n_output;i++){
        output_attrs[i].index = i;
        rknn_query(ctx, RKNN_QUERY_NATIVE_NHWC_OUTPUT_ATTR, &output_attrs[i], sizeof(rknn_tensor_attr));
    }

    app_ctx->rknn_ctx = ctx;
    app_ctx->io_num = io_num;
    app_ctx->input_attrs = input_attrs;
    app_ctx->output_attrs = output_attrs;
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        app_ctx->model_channel = input_attrs[0].dims[1];
        app_ctx->model_height = input_attrs[0].dims[2];
        app_ctx->model_width = input_attrs[0].dims[3];
    } else {
        app_ctx->model_height = input_attrs[0].dims[1];
        app_ctx->model_width = input_attrs[0].dims[2];
        app_ctx->model_channel = input_attrs[0].dims[3];
    }

    // only support float outputs for now
    if (output_attrs[0].qnt_type == RKNN_TENSOR_QNT_AFFINE_ASYMMETRIC && output_attrs[0].type != RKNN_TENSOR_FLOAT16) {
        app_ctx->is_quant = true;
        printf("Warning: model appears quantized. enabling quantized dequantization path.\n");
    } else {
        app_ctx->is_quant = false;
    }

    // Determine fmc, feat strides, num_anchors, use_kps based on output count (mirror python logic)
    int out_cnt = app_ctx->io_num.n_output;
    if (out_cnt == 6) {
        app_ctx->fmc = 3;
        app_ctx->feat_strides_count = 3;
        app_ctx->feat_strides = (int*)malloc(sizeof(int)*3);
        app_ctx->feat_strides[0]=8; app_ctx->feat_strides[1]=16; app_ctx->feat_strides[2]=32;
        app_ctx->num_anchors = 2;
        app_ctx->use_kps = false;
    } else if (out_cnt == 9) {
        app_ctx->fmc = 3;
        app_ctx->feat_strides_count = 3;
        app_ctx->feat_strides = (int*)malloc(sizeof(int)*3);
        app_ctx->feat_strides[0]=8; app_ctx->feat_strides[1]=16; app_ctx->feat_strides[2]=32;
        app_ctx->num_anchors = 2;
        app_ctx->use_kps = true;
    } else if (out_cnt == 10) {
        app_ctx->fmc = 5;
        app_ctx->feat_strides_count = 5;
        app_ctx->feat_strides = (int*)malloc(sizeof(int)*5);
        app_ctx->feat_strides[0]=8; app_ctx->feat_strides[1]=16; app_ctx->feat_strides[2]=32; app_ctx->feat_strides[3]=64; app_ctx->feat_strides[4]=128;
        app_ctx->num_anchors = 1;
        app_ctx->use_kps = false;
    } else if (out_cnt == 15) {
        app_ctx->fmc = 5;
        app_ctx->feat_strides_count = 5;
        app_ctx->feat_strides = (int*)malloc(sizeof(int)*5);
        app_ctx->feat_strides[0]=8; app_ctx->feat_strides[1]=16; app_ctx->feat_strides[2]=32; app_ctx->feat_strides[3]=64; app_ctx->feat_strides[4]=128;
        app_ctx->num_anchors = 1;
        app_ctx->use_kps = true;
    } else {
        // fallback: assume 3 fmc without kps
        app_ctx->fmc = 3;
        app_ctx->feat_strides_count = 3;
        app_ctx->feat_strides = (int*)malloc(sizeof(int)*3);
        app_ctx->feat_strides[0]=8; app_ctx->feat_strides[1]=16; app_ctx->feat_strides[2]=32;
        app_ctx->num_anchors = 2;
        app_ctx->use_kps = false;
        printf("Warning: unexpected output count %d, using default fmc=3, num_anchors=2\n", out_cnt);
    }

    // default nms thresh if not set
    if (app_ctx->nms_thresh <= 0.0f) app_ctx->nms_thresh = 0.4f;

    printf("scrfd model height=%d width=%d channel=%d fmc=%d num_anchors=%d use_kps=%d\n",
           app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel, app_ctx->fmc, app_ctx->num_anchors, app_ctx->use_kps ? 1 : 0);
    return 0;
}

int release_scrfd_model(rknn_app_context_t* app_ctx) {
    if (!app_ctx) return -1;
    if (app_ctx->input_attrs) { free(app_ctx->input_attrs); app_ctx->input_attrs = NULL; }
    if (app_ctx->output_attrs) { free(app_ctx->output_attrs); app_ctx->output_attrs = NULL; }
    if (app_ctx->rknn_ctx != 0) { rknn_destroy(app_ctx->rknn_ctx); app_ctx->rknn_ctx = 0; }
    if (app_ctx->feat_strides) { free(app_ctx->feat_strides); app_ctx->feat_strides = NULL; }
    center_cache.clear();
    return 0;
}

int inference_scrfd_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_result_list* od_results, float conf_thresh) {
    if (!app_ctx || !img || !od_results) return -1;

    int ret;
    image_buffer_t dst_img;
    letterbox_t letter_box;
    memset(&dst_img, 0, sizeof(dst_img));
    memset(&letter_box, 0, sizeof(letterbox_t));

    // Preprocess: letterbox resize
    dst_img.width = app_ctx->model_width;
    dst_img.height = app_ctx->model_height;
    dst_img.format = IMAGE_FORMAT_RGB888;
    dst_img.size = get_image_size(&dst_img);
    dst_img.virt_addr = (unsigned char*)malloc(dst_img.size);
    if (!dst_img.virt_addr) return -1;

    int bg_color = 0;
    ret = convert_image_with_letterbox(img, &dst_img, &letter_box, bg_color);
    if (ret < 0) {
        free(dst_img.virt_addr);
        return -1;
    }

    // Set inputs and run
    rknn_input inputs[1];
    int out_n = app_ctx->io_num.n_output;
    rknn_output* outputs = (rknn_output*)malloc(sizeof(rknn_output) * out_n);
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(rknn_output) * out_n);

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = dst_img.size;
    inputs[0].buf = dst_img.virt_addr;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0) {
        free(dst_img.virt_addr);
        free(outputs);
        return -1;
    }

    ret = rknn_run(app_ctx->rknn_ctx, NULL);
    if (ret < 0) {
        free(dst_img.virt_addr);
        free(outputs);
        return -1;
    }

    // Get outputs (want float for inference)
    for (int i = 0; i < out_n; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, out_n, outputs, NULL);
    if (ret < 0) {
        free(dst_img.virt_addr);
        free(outputs);
        return -1;
    }

    // Collect detections from all feature maps (mirror python forward)
    std::vector<FaceDet> all_dets; // used for NMS
    std::vector<float> all_scores;
    std::vector<std::array<float,4>> all_bboxes_model; // model input coords
    std::vector<std::array<float,10>> all_kps_model;   // flattened kps in model coords (if present)

    int orig_w = img->width, orig_h = img->height;

    int fmc = app_ctx->fmc;
    int num_anchors = app_ctx->num_anchors;
    bool use_kps = app_ctx->use_kps;

    // For each stride index
    for (int si = 0; si < app_ctx->feat_strides_count; si++) {
        int stride = app_ctx->feat_strides[si];
        int feat_h = (int)ceil((float)app_ctx->model_height / stride);
        int feat_w = (int)ceil((float)app_ctx->model_width / stride);
        int N = feat_h * feat_w * num_anchors;

        int score_idx = si;
        int bbox_idx = si + fmc;
        int kps_idx = si + fmc * 2;

        if (score_idx >= out_n || bbox_idx >= out_n) continue;

        float* score_ptr = (float*)outputs[score_idx].buf;
        float* bbox_ptr = (float*)outputs[bbox_idx].buf;
        float* kps_ptr = nullptr;
        if (use_kps) {
            if (kps_idx < out_n) kps_ptr = (float*)outputs[kps_idx].buf;
        }

        if (!score_ptr || !bbox_ptr) continue;

        // Generate anchor centers for this stride (cached)
        std::vector<std::array<float,2>> anchors;
        generate_anchor_centers_cached(anchors, feat_h, feat_w, stride, num_anchors);
        if ((int)anchors.size() < N) {
            // safety
            continue;
        }

        // Decode bboxes and kps for this stride
        std::vector<std::array<float,4>> decoded_bboxes;
        distance2bbox(anchors, bbox_ptr, N, stride, decoded_bboxes);

        std::vector<std::array<float,10>> decoded_kps;
        if (use_kps && kps_ptr) {
            distance2kps(anchors, kps_ptr, N, stride, decoded_kps);
        }

        // Collect positive indices where score >= conf_thresh
        for (int i = 0; i < N; i++) {
            float score = score_ptr[i];
            if (score > 1.0f || score < 0.0f) {
                score = 1.0f / (1.0f + expf(-score));
            }
            if (score < conf_thresh) continue;

            // keep model-space bbox and kps for now; mapping to original image later
            std::array<float,4> mb = decoded_bboxes[i];
            // clip in model space to [0, model_w/ h] to avoid degenerate boxes
            mb[0] = std::max(0.f, std::min(mb[0], (float)app_ctx->model_width));
            mb[1] = std::max(0.f, std::min(mb[1], (float)app_ctx->model_height));
            mb[2] = std::max(0.f, std::min(mb[2], (float)app_ctx->model_width));
            mb[3] = std::max(0.f, std::min(mb[3], (float)app_ctx->model_height));
            if (mb[2] <= mb[0] || mb[3] <= mb[1]) continue;

            all_scores.push_back(score);
            all_bboxes_model.push_back(mb);
            if (use_kps && kps_ptr) {
                all_kps_model.push_back(decoded_kps[i]);
            } else if (use_kps) {
                // if model expects kps but kps_ptr missing, push zeros to keep indexing aligned
                std::array<float,10> zeros = {0};
                all_kps_model.push_back(zeros);
            }
        }
    }

    rknn_outputs_release(app_ctx->rknn_ctx, out_n, outputs);
    free(outputs);
    free(dst_img.virt_addr);

    int M = (int)all_scores.size();
    if (M == 0) {
        od_results->count = 0;
        return 0;
    }

    // Build dets for NMS and preserve mapping to all_* arrays
    std::vector<FaceDet> dets(M);
    for (int i = 0; i < M; i++) {
        dets[i].x1 = all_bboxes_model[i][0];
        dets[i].y1 = all_bboxes_model[i][1];
        dets[i].x2 = all_bboxes_model[i][2];
        dets[i].y2 = all_bboxes_model[i][3];
        dets[i].score = all_scores[i];
    }

    // Run NMS
    auto keep = nms_keep(dets, app_ctx->nms_thresh);

    // Fill results mapping back to original image using letterbox info
    od_results->count = 0;
    for (size_t kidx = 0; kidx < keep.size() && kidx < OBJ_NUMB_MAX_SIZE; ++kidx) {
        int idx = keep[kidx]; // index into all_* arrays

        // Map bbox from model input coords to original image coords
        float x1 = (all_bboxes_model[idx][0] - letter_box.x_pad) / letter_box.scale;
        float y1 = (all_bboxes_model[idx][1] - letter_box.y_pad) / letter_box.scale;
        float x2 = (all_bboxes_model[idx][2] - letter_box.x_pad) / letter_box.scale;
        float y2 = (all_bboxes_model[idx][3] - letter_box.y_pad) / letter_box.scale;

        // Clip to original image bounds
        x1 = std::max(0.f, std::min(x1, (float)orig_w));
        y1 = std::max(0.f, std::min(y1, (float)orig_h));
        x2 = std::max(0.f, std::min(x2, (float)orig_w));
        y2 = std::max(0.f, std::min(y2, (float)orig_h));

        if (x2 <= x1 || y2 <= y1) continue;

        od_results->results[kidx].box.left = (int)roundf(x1);
        od_results->results[kidx].box.top = (int)roundf(y1);
        od_results->results[kidx].box.right = (int)roundf(x2);
        od_results->results[kidx].box.bottom = (int)roundf(y2);
        od_results->results[kidx].prop = dets[idx].score;

        // Map keypoints if present and store into od_results (ensure od_results struct supports kps)
        if (use_kps && idx < (int)all_kps_model.size()) {
            for (int p = 0; p < 5; p++) {
                float kx = (all_kps_model[idx][p*2 + 0] - letter_box.x_pad) / letter_box.scale;
                float ky = (all_kps_model[idx][p*2 + 1] - letter_box.y_pad) / letter_box.scale;
                kx = std::max(0.f, std::min(kx, (float)orig_w));
                ky = std::max(0.f, std::min(ky, (float)orig_h));
                od_results->results[kidx].kps[p].x = (int)roundf(kx);
                od_results->results[kidx].kps[p].y = (int)roundf(ky);            
            }
        }

        od_results->count++;
    }

    return 0;
}

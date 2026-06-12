#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <array>
#include <stdint.h>

#include "scrfd_rknn.h"
#include "file_utils.h"
#include "image_utils.h"

static const int STRIDES[] = {8, 16, 32};
static const int FMC = 3; // feature map count
static const int NUM_ANCHORS = 2;

static void generate_anchor_centers(std::vector<std::array<float,2>>& centers, int feat_h, int feat_w, int stride) {
    centers.clear();
    centers.reserve(feat_h * feat_w * NUM_ANCHORS);
    for (int gy = 0; gy < feat_h; gy++)
        for (int gx = 0; gx < feat_w; gx++)
            for (int a = 0; a < NUM_ANCHORS; a++)
                centers.push_back({(float)(gx * stride), (float)(gy * stride)});
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

struct FaceDet { float x1,y1,x2,y2,score; float kps[5][2]; };

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

    printf("scrfd model height=%d width=%d channel=%d\n", app_ctx->model_height, app_ctx->model_width, app_ctx->model_channel);
    return 0;
}

int release_scrfd_model(rknn_app_context_t* app_ctx) {
    if (!app_ctx) return -1;
    if (app_ctx->input_attrs) { free(app_ctx->input_attrs); app_ctx->input_attrs = NULL; }
    if (app_ctx->output_attrs) { free(app_ctx->output_attrs); app_ctx->output_attrs = NULL; }
    if (app_ctx->rknn_ctx != 0) { rknn_destroy(app_ctx->rknn_ctx); app_ctx->rknn_ctx = 0; }
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
    rknn_output outputs[app_ctx->io_num.n_output];
    memset(inputs, 0, sizeof(inputs));
    memset(outputs, 0, sizeof(outputs));

    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = dst_img.size;
    inputs[0].buf = dst_img.virt_addr;

    ret = rknn_inputs_set(app_ctx->rknn_ctx, 1, inputs);
    if (ret < 0) {
        free(dst_img.virt_addr);
        return -1;
    }

    ret = rknn_run(app_ctx->rknn_ctx, NULL);
    if (ret < 0) {
        free(dst_img.virt_addr);
        return -1;
    }

    // Get outputs (want float for inference)
    for (int i = 0; i < app_ctx->io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = 1;
    }
    ret = rknn_outputs_get(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs, NULL);
    if (ret < 0) {
        free(dst_img.virt_addr);
        return -1;
    }

    // Collect detections from all feature maps
    std::vector<FaceDet> all_dets;
    int orig_w = img->width, orig_h = img->height;

    // Parse outputs for each stride: outputs[0-2] are scores, [3-5] are bboxes, [6-8] are keypoints
    for (int si = 0; si < FMC; si++) {
        int stride = STRIDES[si];
        int feat_h = (int)ceil((float)app_ctx->model_height / stride);
        int feat_w = (int)ceil((float)app_ctx->model_width / stride);
        int N = feat_h * feat_w * NUM_ANCHORS;

        // Get pointers to score, bbox, and keypoint outputs
        float* score_ptr = (float*)outputs[si].buf;          // stride 8/16/32 scores
        float* bbox_ptr = (float*)outputs[si + FMC].buf;     // stride 8/16/32 bboxes
        float* kps_ptr  = (float*)outputs[si + FMC*2].buf;   // stride 8/16/32 keypoints
        
        if (!score_ptr || !bbox_ptr || !kps_ptr) continue;

        // Generate anchor centers for this stride
        std::vector<std::array<float, 2>> anchors;
        generate_anchor_centers(anchors, feat_h, feat_w, stride);

        // Decode detections
        for (int i = 0; i < N; i++) {
            float score = score_ptr[i];
            
            // Apply sigmoid if score is outside [0,1]
            if (score > 1.0f || score < 0.0f) {
                score = 1.0f / (1.0f + expf(-score));
            }
            
            if (score < conf_thresh) continue;

            // Distance-based box decoding: (cx - l*stride, cy - t*stride, cx + r*stride, cy + b*stride)
            float cx = anchors[i][0];
            float cy = anchors[i][1];
            float x1 = (cx - bbox_ptr[i * 4 + 0] * stride - letter_box.x_pad) / letter_box.scale;
            float y1 = (cy - bbox_ptr[i * 4 + 1] * stride - letter_box.y_pad) / letter_box.scale;
            float x2 = (cx + bbox_ptr[i * 4 + 2] * stride - letter_box.x_pad) / letter_box.scale;
            float y2 = (cy + bbox_ptr[i * 4 + 3] * stride - letter_box.y_pad) / letter_box.scale;

            // Clip to image bounds
            x1 = std::max(0.f, std::min(x1, (float)orig_w));
            y1 = std::max(0.f, std::min(y1, (float)orig_h));
            x2 = std::max(0.f, std::min(x2, (float)orig_w));
            y2 = std::max(0.f, std::min(y2, (float)orig_h));

            if (x2 <= x1 || y2 <= y1) continue;

            FaceDet det;
            det.x1 = x1;
            det.y1 = y1;
            det.x2 = x2;
            det.y2 = y2;
            det.score = score;
            for (int k = 0; k < 5; k++) {
                det.kps[k][0] = (cx + kps_ptr[i * 10 + k * 2 + 0] * stride - letter_box.x_pad) / letter_box.scale;
                det.kps[k][1] = (cy + kps_ptr[i * 10 + k * 2 + 1] * stride - letter_box.y_pad) / letter_box.scale;
                det.kps[k][0] = std::max(0.f, std::min(det.kps[k][0], (float)orig_w));
                det.kps[k][1] = std::max(0.f, std::min(det.kps[k][1], (float)orig_h));
            }
            all_dets.push_back(det);
        }
    }

    rknn_outputs_release(app_ctx->rknn_ctx, app_ctx->io_num.n_output, outputs);
    free(dst_img.virt_addr);

    if (all_dets.empty()) {
        od_results->count = 0;
        return 0;
    }

    // Apply NMS
    auto keep = nms_keep(all_dets, 0.4f);

    // Fill results
    od_results->count = 0;
    for (size_t i = 0; i < keep.size() && i < OBJ_NUMB_MAX_SIZE; i++) {
        int idx = keep[i];
        od_results->results[i].box.left = (int)all_dets[idx].x1;
        od_results->results[i].box.top = (int)all_dets[idx].y1;
        od_results->results[i].box.right = (int)all_dets[idx].x2;
        od_results->results[i].box.bottom = (int)all_dets[idx].y2;
        od_results->results[i].prop = all_dets[idx].score;
        od_results->results[i].cls_id = 0;
        for (int k = 0; k < 5; k++) {
            od_results->results[i].kps[k][0] = all_dets[idx].kps[k][0];
            od_results->results[i].kps[k][1] = all_dets[idx].kps[k][1];
        }
        od_results->count++;
    }

    return 0;
}

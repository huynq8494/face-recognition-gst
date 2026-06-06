#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <numeric>
#include <cmath>

// SCRFD config: use_kps=true, 9 outputs, fmc=3, strides=[8,16,32], num_anchors=2
static const std::vector<int> FEAT_STRIDE_FPN = {8, 16, 32};
static const int   NUM_ANCHORS = 2;
static const float NMS_THRESH  = 0.4f;
static const float CONF_THRESH = 0.5f;

struct FaceDetection {
    float x1, y1, x2, y2, score;
    float kps[5][2]; // 5 facial keypoints (x,y)
};

struct ResizeInfo {
    float scale;
    int   new_w, new_h;
};

// Preprocess: resize keeping aspect ratio (top-left placement), normalize (x-127.5)/128, BGR->RGB, CHW
std::vector<float> preprocess(const cv::Mat& img, const cv::Size& input_size, ResizeInfo& ri) {
    float sx = (float)input_size.width  / img.cols;
    float sy = (float)input_size.height / img.rows;
    ri.scale = std::min(sx, sy);
    ri.new_w = (int)(img.cols * ri.scale);
    ri.new_h = (int)(img.rows * ri.scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(ri.new_w, ri.new_h));

    // Zero-padded canvas, top-left placement (matches Python)
    cv::Mat padded(input_size.height, input_size.width, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(padded(cv::Rect(0, 0, ri.new_w, ri.new_h)));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F);

    // CHW, normalize: (pixel - 127.5) / 128
    std::vector<cv::Mat> chans(3);
    cv::split(rgb, chans);
    std::vector<float> tensor;
    tensor.reserve(3 * input_size.height * input_size.width);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < input_size.height; y++)
            for (int x = 0; x < input_size.width; x++)
                tensor.push_back((chans[c].at<float>(y, x) - 127.5f) / 128.0f);
    return tensor;
}

// Generate anchor centers [H*W*num_anchors, 2] for a given feature level
// Each (gx, gy) cell gets num_anchors identical centers at (gx*stride, gy*stride)
std::vector<std::array<float, 2>> generateAnchorCenters(int feat_h, int feat_w,
                                                         int stride, int num_anchors) {
    std::vector<std::array<float, 2>> centers;
    centers.reserve(feat_h * feat_w * num_anchors);
    for (int gy = 0; gy < feat_h; gy++)
        for (int gx = 0; gx < feat_w; gx++)
            for (int a = 0; a < num_anchors; a++)
                centers.push_back({ (float)(gx * stride), (float)(gy * stride) });
    return centers;
}

// Standard IoU-based NMS (same logic as Python nms())
std::vector<int> nms(const std::vector<FaceDetection>& dets, float thresh) {
    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b){ return dets[a].score > dets[b].score; });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<int>  keep;
    for (int i = 0; i < (int)order.size(); i++) {
        int ii = order[i];
        if (suppressed[ii]) continue;
        keep.push_back(ii);
        const auto& d = dets[ii];
        float area_i = (d.x2 - d.x1 + 1) * (d.y2 - d.y1 + 1);
        for (int j = i + 1; j < (int)order.size(); j++) {
            int jj = order[j];
            if (suppressed[jj]) continue;
            const auto& e = dets[jj];
            float ix1  = std::max(d.x1, e.x1), iy1 = std::max(d.y1, e.y1);
            float ix2  = std::min(d.x2, e.x2), iy2 = std::min(d.y2, e.y2);
            float inter = std::max(0.f, ix2 - ix1 + 1) * std::max(0.f, iy2 - iy1 + 1);
            float area_j = (e.x2 - e.x1 + 1) * (e.y2 - e.y1 + 1);
            float ovr = inter / (area_i + area_j - inter + 1e-6f);
            if (ovr > thresh) suppressed[jj] = true;
        }
    }
    return keep;
}

int main() {
    // ── Session ────────────────────────────────────────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "scrfd");
    Ort::SessionOptions opts;
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    Ort::Session session(env,
        "/home/huynq/models/scrfd/scrfd_500m_bnkps_shape640x640.onnx", opts);

    const cv::Size input_size(640, 640); // (width, height)

    // ── Load image ─────────────────────────────────────────────────────────────
    cv::Mat img = cv::imread("/home/huynq/data/images/6faces.jpg");
    if (img.empty()) { std::cerr << "Failed to load image\n"; return -1; }
    int orig_h = img.rows, orig_w = img.cols;
    std::cout << "Image: " << orig_w << "x" << orig_h << "\n";

    // ── Preprocess ─────────────────────────────────────────────────────────────
    ResizeInfo ri;
    auto blob = preprocess(img, input_size, ri);
    std::cout << "Scale: " << ri.scale
              << "  resized: " << ri.new_w << "x" << ri.new_h << "\n";

    std::array<int64_t, 4> in_shape{1, 3, input_size.height, input_size.width};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem, blob.data(), blob.size(), in_shape.data(), in_shape.size());

    // ── Output names ───────────────────────────────────────────────────────────
    Ort::AllocatorWithDefaultOptions allocator;
    size_t n_out = session.GetOutputCount();
    std::cout << "Number of outputs: " << n_out << "\n"; // expect 9 for use_kps
    std::vector<const char*>              out_names;
    std::vector<Ort::AllocatedStringPtr>  out_name_allocs;
    for (size_t i = 0; i < n_out; i++) {
        out_name_allocs.push_back(session.GetOutputNameAllocated(i, allocator));
        out_names.push_back(out_name_allocs.back().get());
    }

    auto in_name_alloc = session.GetInputNameAllocated(0, allocator);
    const char* in_name = in_name_alloc.get();

    // ── Inference ──────────────────────────────────────────────────────────────
    auto out_tensors = session.Run(
        Ort::RunOptions{nullptr}, &in_name, &in_tensor, 1,
        out_names.data(), out_names.size());

    // ── Postprocess ────────────────────────────────────────────────────────────
    // Output layout (9 tensors, use_kps=true, fmc=3):
    //   out[0..2]  : score  [N, 1]   for stride 8, 16, 32
    //   out[3..5]  : bbox   [N, 4]   distance (l,t,r,b) before ×stride
    //   out[6..8]  : kps    [N, 10]  offsets for 5 keypoints before ×stride
    const int fmc = 3;
    std::vector<FaceDetection> all_dets;

    for (int si = 0; si < fmc; si++) {
        int stride = FEAT_STRIDE_FPN[si];
        int feat_h = (int)std::ceil((float)input_size.height / stride);
        int feat_w = (int)std::ceil((float)input_size.width  / stride);
        int N      = feat_h * feat_w * NUM_ANCHORS;

        float* score_ptr = out_tensors[si       ].GetTensorMutableData<float>(); // [N,1]
        float* bbox_ptr  = out_tensors[si + fmc  ].GetTensorMutableData<float>(); // [N,4]
        float* kps_ptr   = out_tensors[si + fmc*2].GetTensorMutableData<float>(); // [N,10]

        auto anchors = generateAnchorCenters(feat_h, feat_w, stride, NUM_ANCHORS);

        for (int i = 0; i < N; i++) {
            float score = score_ptr[i]; // shape [N,1] → element i
            if (score < CONF_THRESH) continue;

            float cx = anchors[i][0];
            float cy = anchors[i][1];

            // distance2bbox: x1=cx-l, y1=cy-t, x2=cx+r, y2=cy+b  (×stride already baked in)
            float x1 = cx - bbox_ptr[i*4 + 0] * stride;
            float y1 = cy - bbox_ptr[i*4 + 1] * stride;
            float x2 = cx + bbox_ptr[i*4 + 2] * stride;
            float y2 = cy + bbox_ptr[i*4 + 3] * stride;

            // Rescale to original image space
            x1 = std::max(0.f, std::min(x1 / ri.scale, (float)orig_w));
            y1 = std::max(0.f, std::min(y1 / ri.scale, (float)orig_h));
            x2 = std::max(0.f, std::min(x2 / ri.scale, (float)orig_w));
            y2 = std::max(0.f, std::min(y2 / ri.scale, (float)orig_h));
            if (x2 <= x1 || y2 <= y1) continue;

            FaceDetection det;
            det.x1 = x1; det.y1 = y1; det.x2 = x2; det.y2 = y2;
            det.score = score;

            // distance2kps: kx=cx+dx*stride, ky=cy+dy*stride
            for (int k = 0; k < 5; k++) {
                det.kps[k][0] = (cx + kps_ptr[i*10 + k*2 + 0] * stride) / ri.scale;
                det.kps[k][1] = (cy + kps_ptr[i*10 + k*2 + 1] * stride) / ri.scale;
            }
            all_dets.push_back(det);
        }
    }

    auto keep = nms(all_dets, NMS_THRESH);
    std::cout << "Raw detections : " << all_dets.size() << "\n";
    std::cout << "After NMS      : " << keep.size()     << "\n";

    // ── Draw ───────────────────────────────────────────────────────────────────
    cv::Mat result = img.clone();
    for (int idx : keep) {
        const auto& d = all_dets[idx];
        std::cout << "[DET] score=" << d.score
                  << " bbox=(" << d.x1 << "," << d.y1 << ")-(" << d.x2 << "," << d.y2 << ")"
                  << " size=" << (d.x2-d.x1) << "x" << (d.y2-d.y1) << "\n";

        cv::rectangle(result,
            cv::Point((int)d.x1, (int)d.y1),
            cv::Point((int)d.x2, (int)d.y2),
            cv::Scalar(255, 0, 0), 2);

        // Draw 5 keypoints
        for (int k = 0; k < 5; k++)
            cv::circle(result,
                cv::Point((int)d.kps[k][0], (int)d.kps[k][1]),
                2, cv::Scalar(0, 0, 255), 2);

        std::string label = "Face " + std::to_string((int)(d.score * 100)) + "%";
        cv::putText(result, label,
            cv::Point((int)d.x1, std::max((int)d.y1 - 4, 12)),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
    }

    cv::imshow("SCRFD", result);
    cv::imwrite("/tmp/scrfd_result.jpg", result);
    cv::waitKey(0);
    return 0;
}

#include "scrfd.hpp"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <iostream>

const std::vector<int> SCRFD::STRIDES = {8, 16, 32};

SCRFD::SCRFD(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "scrfd"),
      session_(env_, model_path.c_str(), Ort::SessionOptions{})
{
    auto in_alloc   = session_.GetInputNameAllocated(0, allocator_);
    input_name_     = in_alloc.get();

    size_t n = session_.GetOutputCount();
    std::cout << "[SCRFD] Outputs: " << n << "\n";
    for (size_t i = 0; i < n; i++) {
        auto alloc = session_.GetOutputNameAllocated(i, allocator_);
        output_names_str_.push_back(alloc.get());
    }
    for (auto& s : output_names_str_)
        output_names_.push_back(s.c_str());
}

std::vector<float> SCRFD::preprocess(const cv::Mat& img, const cv::Size& input_size,
                                      float& scale, int& new_w, int& new_h) {
    float sx = (float)input_size.width  / img.cols;
    float sy = (float)input_size.height / img.rows;
    scale = std::min(sx, sy);
    new_w = (int)(img.cols * scale);
    new_h = (int)(img.rows * scale);

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_w, new_h));

    // Zero-padded canvas, top-left placement
    cv::Mat padded(input_size.height, input_size.width, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(padded(cv::Rect(0, 0, new_w, new_h)));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F);

    // CHW + normalize: (pixel - 127.5) / 128
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

std::vector<std::array<float, 2>>
SCRFD::generateAnchorCenters(int feat_h, int feat_w, int stride) {
    std::vector<std::array<float, 2>> centers;
    centers.reserve(feat_h * feat_w * NUM_ANCHORS);
    for (int gy = 0; gy < feat_h; gy++)
        for (int gx = 0; gx < feat_w; gx++)
            for (int a = 0; a < NUM_ANCHORS; a++)
                centers.push_back({(float)(gx * stride), (float)(gy * stride)});
    return centers;
}

std::vector<int> SCRFD::nms(const std::vector<FaceDetection>& dets) {
    std::vector<int> order(dets.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return dets[a].score > dets[b].score; });

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
            const auto& e   = dets[jj];
            float ix1       = std::max(d.x1, e.x1), iy1 = std::max(d.y1, e.y1);
            float ix2       = std::min(d.x2, e.x2), iy2 = std::min(d.y2, e.y2);
            float inter     = std::max(0.f, ix2 - ix1 + 1) * std::max(0.f, iy2 - iy1 + 1);
            float area_j    = (e.x2 - e.x1 + 1) * (e.y2 - e.y1 + 1);
            float ovr       = inter / (area_i + area_j - inter + 1e-6f);
            if (ovr > nms_thresh) suppressed[jj] = true;
        }
    }
    return keep;
}

void SCRFD::detect(const cv::Mat& img, std::vector<FaceDetection>& out_dets,
                   float thresh, const cv::Size& input_size) {
    out_dets.clear();

    float scale; int new_w, new_h;
    auto blob = preprocess(img, input_size, scale, new_w, new_h);

    std::array<int64_t, 4> in_shape{1, 3, input_size.height, input_size.width};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem, blob.data(), blob.size(), in_shape.data(), in_shape.size());

    const char* in_name = input_name_.c_str();
    auto out_tensors = session_.Run(
        Ort::RunOptions{nullptr}, &in_name, &in_tensor, 1,
        output_names_.data(), output_names_.size());

    int orig_w = img.cols, orig_h = img.rows;
    std::vector<FaceDetection> all_dets;

    // Output layout (9 tensors, use_kps=true, fmc=3):
    //   [0..2]  : score [N, 1]   stride 8, 16, 32
    //   [3..5]  : bbox  [N, 4]   distance (l,t,r,b)
    //   [6..8]  : kps   [N, 10]  offsets for 5 keypoints
    for (int si = 0; si < FMC; si++) {
        int stride = STRIDES[si];
        int feat_h = (int)std::ceil((float)input_size.height / stride);
        int feat_w = (int)std::ceil((float)input_size.width  / stride);
        int N      = feat_h * feat_w * NUM_ANCHORS;

        float* score_ptr = out_tensors[si         ].GetTensorMutableData<float>();
        float* bbox_ptr  = out_tensors[si + FMC   ].GetTensorMutableData<float>();
        float* kps_ptr   = out_tensors[si + FMC*2 ].GetTensorMutableData<float>();

        auto anchors = generateAnchorCenters(feat_h, feat_w, stride);

        for (int i = 0; i < N; i++) {
            float score = score_ptr[i];
            if (score < thresh) continue;

            float cx = anchors[i][0], cy = anchors[i][1];

            // distance2bbox: (cx-l, cy-t, cx+r, cy+b) × stride
            float x1 = (cx - bbox_ptr[i*4+0] * stride) / scale;
            float y1 = (cy - bbox_ptr[i*4+1] * stride) / scale;
            float x2 = (cx + bbox_ptr[i*4+2] * stride) / scale;
            float y2 = (cy + bbox_ptr[i*4+3] * stride) / scale;

            x1 = std::max(0.f, std::min(x1, (float)orig_w));
            y1 = std::max(0.f, std::min(y1, (float)orig_h));
            x2 = std::max(0.f, std::min(x2, (float)orig_w));
            y2 = std::max(0.f, std::min(y2, (float)orig_h));
            if (x2 <= x1 || y2 <= y1) continue;

            FaceDetection det;
            det.x1 = x1; det.y1 = y1; det.x2 = x2; det.y2 = y2;
            det.score = score;

            // distance2kps: kx = cx + dx*stride, ky = cy + dy*stride
            for (int k = 0; k < 5; k++) {
                det.kps[k][0] = (cx + kps_ptr[i*10 + k*2+0] * stride) / scale;
                det.kps[k][1] = (cy + kps_ptr[i*10 + k*2+1] * stride) / scale;
            }
            all_dets.push_back(det);
        }
    }

    auto keep = nms(all_dets);
    for (int idx : keep)
        out_dets.push_back(all_dets[idx]);
}

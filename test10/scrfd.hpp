#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "utils.hpp"

// RKNN SCRFD wrapper (local)
#include "rknpu2/scrfd_rknn.h"

class SCRFD {
public:
    explicit SCRFD(const std::string& model_path);
    ~SCRFD();

    // Run detection on BGR `img`. Results in `out_dets` (image coordinates).
    void detect(const cv::Mat& img,
                std::vector<FaceDetection>& out_dets,
                float thresh = 0.5f,
                const cv::Size& input_size = cv::Size(640, 640));

    float nms_thresh = 0.4f;

private:
    rknn_app_context_t rknn_ctx_;
};

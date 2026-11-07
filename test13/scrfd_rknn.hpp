#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "utils.hpp"

// RKNN-based SCRFD face detector
// Wrapper around RKNN C API for C++ usage, compatible with test08 interface
class SCRFD_RKNN {
public:
    explicit SCRFD_RKNN(const std::string& model_path);
    ~SCRFD_RKNN();

    // Main detection API (compatible with test08 SCRFD interface)
    void detect(const cv::Mat& img,
                std::vector<FaceDetection>& out_dets,
                float thresh = 0.5f,
                const cv::Size& input_size = cv::Size(640, 640));

    float nms_thresh = 0.4f;

private:
    void* app_ctx_;  // rknn_app_context_t* (void* to avoid exposing C API)
    
    // Helper to generate estimated 5 keypoints from bbox
    static void generateKeypoints(const cv::Mat& img, 
                                   float x1, float y1, float x2, float y2,
                                   float kps[5][2]);
};

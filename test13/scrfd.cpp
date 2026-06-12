#include "scrfd.hpp"
#include "rknpu2/scrfd_rknn.h"
#include "image_utils.h"
#include <cstring>
#include <stdexcept>
#include <iostream>

SCRFD_RKNN::SCRFD_RKNN(const std::string& model_path) {
    app_ctx_ = new rknn_app_context_t;
    memset(app_ctx_, 0, sizeof(rknn_app_context_t));
    
    int ret = init_scrfd_model(model_path.c_str(), (rknn_app_context_t*)app_ctx_);
    if (ret != 0) {
        delete (rknn_app_context_t*)app_ctx_;
        app_ctx_ = nullptr;
        throw std::runtime_error("Failed to initialize SCRFD RKNN model: " + 
                                 std::to_string(ret));
    }
    std::cout << "[SCRFD_RKNN] Model loaded: " << model_path << "\n";
}

SCRFD_RKNN::~SCRFD_RKNN() {
    if (app_ctx_) {
        release_scrfd_model((rknn_app_context_t*)app_ctx_);
        delete (rknn_app_context_t*)app_ctx_;
        app_ctx_ = nullptr;
    }
}

void SCRFD_RKNN::generateKeypoints(const cv::Mat& img, 
                                    float x1, float y1, float x2, float y2,
                                    float kps[5][2]) {
    // Generate 5 facial keypoints from bbox (approximation)
    // Reference: 5-point landmarks (eyes, nose, mouth corners)
    float w = x2 - x1;
    float h = y2 - y1;
    
    // Eye locations (approximate)
    kps[0][0] = x1 + w * 0.35f;  // Left eye
    kps[0][1] = y1 + h * 0.35f;
    
    kps[1][0] = x1 + w * 0.65f;  // Right eye
    kps[1][1] = y1 + h * 0.35f;
    
    // Nose
    kps[2][0] = x1 + w * 0.50f;
    kps[2][1] = y1 + h * 0.60f;
    
    // Mouth corners
    kps[3][0] = x1 + w * 0.35f;  // Left mouth
    kps[3][1] = y1 + h * 0.80f;
    
    kps[4][0] = x1 + w * 0.65f;  // Right mouth
    kps[4][1] = y1 + h * 0.80f;
}

void SCRFD_RKNN::detect(const cv::Mat& img,
                         std::vector<FaceDetection>& out_dets,
                         float thresh,
                         const cv::Size& input_size) {
    out_dets.clear();
    
    if (!app_ctx_) {
        throw std::runtime_error("SCRFD model not initialized");
    }
    
    // Convert OpenCV Mat to image_buffer_t
    cv::Mat rgb_img;
    if (img.channels() == 3) {
        cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
    } else if (img.channels() == 1) {
        cv::cvtColor(img, rgb_img, cv::COLOR_GRAY2RGB);
    } else {
        rgb_img = img.clone();
    }
    
    // Create image buffer
    image_buffer_t src_img;
    memset(&src_img, 0, sizeof(image_buffer_t));
    src_img.width = rgb_img.cols;
    src_img.height = rgb_img.rows;
    src_img.format = IMAGE_FORMAT_RGB888;
    src_img.size = rgb_img.total() * rgb_img.elemSize();
    src_img.virt_addr = rgb_img.data;
    
    // Run inference
    object_detect_result_list od_results;
    memset(&od_results, 0, sizeof(od_results));
    
    int ret = inference_scrfd_model((rknn_app_context_t*)app_ctx_, 
                                     &src_img, &od_results, thresh);
    if (ret != 0) {
        std::cerr << "Inference failed: " << ret << "\n";
        return;
    }
    
    // Convert results to FaceDetection format
    for (int i = 0; i < od_results.count; i++) {
        FaceDetection det;
        det.x1 = (float)od_results.results[i].box.left;
        det.y1 = (float)od_results.results[i].box.top;
        det.x2 = (float)od_results.results[i].box.right;
        det.y2 = (float)od_results.results[i].box.bottom;
        det.score = od_results.results[i].prop;
        
        // Generate approximate keypoints from bbox
        generateKeypoints(img, det.x1, det.y1, det.x2, det.y2, det.kps);
        
        out_dets.push_back(det);
    }
}

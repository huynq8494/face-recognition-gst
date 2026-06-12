#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "utils.hpp"

/**
 * RKNN-based SCRFD (RKNN2 hardware accelerated)
 * Provides same interface as ONNX SCRFD but with hardware acceleration
 */
class SCRFDRKNN {
public:
    explicit SCRFDRKNN(const std::string& model_path);
    ~SCRFDRKNN();

    // Main detection interface (compatible with ONNX version)
    void detect(const cv::Mat& img,
                std::vector<FaceDetection>& out_dets,
                float thresh = 0.5f,
                const cv::Size& input_size = cv::Size(640, 640));

    float nms_thresh = 0.4f;

private:
    // RKNN context structures
    typedef struct {
        void* rknn_ctx;
        int   io_num_inputs;
        int   io_num_outputs;
        void* input_attrs;   // rknn_tensor_attr*
        void* output_attrs;  // rknn_tensor_attr*
        int   model_channel;
        int   model_width;
        int   model_height;
        bool  is_quant;
    } rknn_context_t;

    rknn_context_t ctx_;
    bool initialized_ = false;

    // Initialization
    int initModel(const char* model_path);
    void releaseModel();

    // RKNN inference wrapper
    int runInference(const cv::Mat& img, const cv::Size& input_size,
                     std::vector<FaceDetection>& dets, float conf_thresh);

    // Utility functions
    static void generateAnchorCenters(std::vector<std::array<float, 2>>& centers,
                                      int feat_h, int feat_w, int stride);
    static std::vector<int> nms(const std::vector<FaceDetection>& dets,
                                float nms_thresh);
};

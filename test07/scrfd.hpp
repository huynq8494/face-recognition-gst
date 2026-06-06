#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "utils.hpp"

class SCRFD {
public:
    explicit SCRFD(const std::string& model_path);

    void detect(const cv::Mat& img,
                std::vector<FaceDetection>& out_dets,
                float thresh = 0.5f,
                const cv::Size& input_size = cv::Size(640, 640));

    float nms_thresh = 0.4f;

private:
    Ort::Env                         env_;
    Ort::Session                     session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::string              input_name_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> output_names_;

    static const std::vector<int> STRIDES;
    static const int NUM_ANCHORS = 2;
    static const int FMC         = 3;

    std::vector<float> preprocess(const cv::Mat& img, const cv::Size& input_size,
                                   float& scale, int& new_w, int& new_h);

    std::vector<std::array<float, 2>> generateAnchorCenters(int feat_h, int feat_w,
                                                             int stride);

    std::vector<int> nms(const std::vector<FaceDetection>& dets);
};

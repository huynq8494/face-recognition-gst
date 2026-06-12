#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include "utils.hpp"

class EdgeFace {
public:
    explicit EdgeFace(const std::string& model_path,
                      const cv::Size& input_size = cv::Size(112, 112));

    // Extract embedding from pre-cropped/aligned face image
    std::vector<float> getEmbedding(const cv::Mat& face_img);

    // Extract embedding from full image + detection (aligns using kps)
    std::vector<float> getEmbeddingFromDet(const cv::Mat& img,
                                            const FaceDetection& det);

    // Align face using 5 keypoints → affine warp
    static cv::Mat alignFace(const cv::Mat& img, const float kps[5][2],
                              const cv::Size& output_size = cv::Size(112, 112));

    // Crop face from bbox when kps unavailable
    static cv::Mat cropFace(const cv::Mat& img, const FaceDetection& det,
                             float margin = 0.0f);

    static float cosineSimilarity(const std::vector<float>& e1,
                                   const std::vector<float>& e2);

    int embeddingSize() const { return embedding_size_; }

private:
    Ort::Env                         env_;
    Ort::Session                     session_;
    Ort::AllocatorWithDefaultOptions allocator_;

    std::string              input_name_;
    std::vector<std::string> output_names_str_;
    std::vector<const char*> output_names_;
    cv::Size                 input_size_;
    int                      embedding_size_ = 512;

    std::vector<float> preprocess(const cv::Mat& face_img);
    static void        l2Normalize(std::vector<float>& v);
};

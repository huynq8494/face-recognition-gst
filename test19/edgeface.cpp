#include "edgeface.hpp"
#include "utils.hpp"
#include <cmath>
#include <iostream>

EdgeFace::EdgeFace(const std::string& model_path, const cv::Size& input_size)
    : env_(ORT_LOGGING_LEVEL_WARNING, "edgeface"),
      session_(env_, model_path.c_str(), Ort::SessionOptions{}),
      input_size_(input_size)
{
    auto in_alloc = session_.GetInputNameAllocated(0, allocator_);
    input_name_   = in_alloc.get();

    size_t n = session_.GetOutputCount();
    for (size_t i = 0; i < n; i++) {
        auto alloc = session_.GetOutputNameAllocated(i, allocator_);
        output_names_str_.push_back(alloc.get());
    }
    for (auto& s : output_names_str_)
        output_names_.push_back(s.c_str());

    // Infer embedding size from model output shape
    auto shape     = session_.GetOutputTypeInfo(0)
                             .GetTensorTypeAndShapeInfo().GetShape();
    embedding_size_ = (shape.back() > 0) ? (int)shape.back() : 512;
    std::cout << "[EdgeFace] Embedding size: " << embedding_size_ << "\n";
}

cv::Mat EdgeFace::alignFace(const cv::Mat& img, const float kps[5][2],
                              const cv::Size& output_size) {
    std::vector<cv::Point2f> src(5), dst(5);
    for (int i = 0; i < 5; i++) {
        src[i] = cv::Point2f(kps[i][0], kps[i][1]);
        dst[i] = cv::Point2f(
            FACE_ALIGN_REF_POINTS[i][0] * output_size.width  / 112.0f,
            FACE_ALIGN_REF_POINTS[i][1] * output_size.height / 112.0f);
    }
    cv::Mat M = cv::estimateAffinePartial2D(src, dst, cv::noArray(), cv::LMEDS);
    cv::Mat aligned;
    cv::warpAffine(img, aligned, M, output_size, cv::INTER_LINEAR);
    return aligned;
}

cv::Mat EdgeFace::cropFace(const cv::Mat& img, const FaceDetection& det, float margin) {
    int x1 = (int)det.x1, y1 = (int)det.y1;
    int x2 = (int)det.x2, y2 = (int)det.y2;
    int w = x2 - x1,  h = y2 - y1;
    x1 = std::max(0,          (int)(x1 - w * margin));
    y1 = std::max(0,          (int)(y1 - h * margin));
    x2 = std::min(img.cols,   (int)(x2 + w * margin));
    y2 = std::min(img.rows,   (int)(y2 + h * margin));
    return img(cv::Rect(x1, y1, x2-x1, y2-y1)).clone();
}

std::vector<float> EdgeFace::preprocess(const cv::Mat& face_img) {
    cv::Mat resized;
    cv::resize(face_img, resized, input_size_);

    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F);

    std::vector<cv::Mat> chans(3);
    cv::split(rgb, chans);
    std::vector<float> tensor;
    tensor.reserve(3 * input_size_.height * input_size_.width);
    for (int c = 0; c < 3; c++)
        for (int y = 0; y < input_size_.height; y++)
            for (int x = 0; x < input_size_.width; x++)
                tensor.push_back((chans[c].at<float>(y, x) - 127.5f) / 128.0f);
    return tensor;
}

void EdgeFace::l2Normalize(std::vector<float>& v) {
    float norm = 0.f;
    for (float f : v) norm += f * f;
    norm = std::sqrt(norm);
    if (norm > 1e-10f)
        for (float& f : v) f /= norm;
}

std::vector<float> EdgeFace::getEmbedding(const cv::Mat& face_img) {
    auto blob = preprocess(face_img);

    std::array<int64_t, 4> in_shape{1, 3, input_size_.height, input_size_.width};
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
        mem, blob.data(), blob.size(), in_shape.data(), in_shape.size());

    const char* in_name = input_name_.c_str();
    auto out = session_.Run(
        Ort::RunOptions{nullptr}, &in_name, &in_tensor, 1,
        output_names_.data(), output_names_.size());

    // Get actual size from output tensor (handles dynamic shapes)
    auto out_shape  = out[0].GetTensorTypeAndShapeInfo().GetShape();
    int  actual_dim = (int)out_shape.back();
    if (actual_dim <= 0) actual_dim = embedding_size_; // fallback
    embedding_size_ = actual_dim;                      // update for next call

    float* data = out[0].GetTensorMutableData<float>();
    std::vector<float> emb(data, data + embedding_size_);
    l2Normalize(emb);
    return emb;
}


std::vector<float> EdgeFace::getEmbeddingFromDet(const cv::Mat& img,
                                                   const FaceDetection& det) {
    cv::Mat face = alignFace(img, det.kps, input_size_);
    return getEmbedding(face);
}

float EdgeFace::cosineSimilarity(const std::vector<float>& e1,
                                  const std::vector<float>& e2) {
    float dot = 0.f;
    for (size_t i = 0; i < e1.size(); i++) dot += e1[i] * e2[i];
    return dot; // works on L2-normalised vectors
}

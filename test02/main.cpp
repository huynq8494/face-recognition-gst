#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <array>

// Preprocess: resize, transpose (HWC->CHW), normalize
std::vector<float> preprocess(const cv::Mat& img) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(640, 640));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<float> input_tensor_values;
    input_tensor_values.reserve(1 * 3 * 640 * 640);

    // HWC -> CHW
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 640; y++) {
            for (int x = 0; x < 640; x++) {
                input_tensor_values.push_back(resized.at<cv::Vec3f>(y, x)[c]);
            }
        }
    }
    return input_tensor_values;
}

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    Ort::Session session(env, "/home/huynq/models/yolo/yolo26n.onnx", session_options);
    cv::Mat img = cv::imread("/home/huynq/data/images/baggage_claim.jpg");
    int h = img.rows, w = img.cols;

    auto input_tensor_values = preprocess(img);
    std::array<int64_t, 4> input_shape{ 1, 3, 640, 640 };

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);

    const char* input_name = input_name_alloc.get();
    const char* output_name = output_name_alloc.get();

    auto output_tensors = session.Run(
        Ort::RunOptions{ nullptr },
        &input_name, &input_tensor, 1,
        &output_name, 1);

    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

    std::cout << "Output shape: "; // 1x300x6
    for (auto d : output_shape) std::cout << d << " ";
    std::cout << std::endl;

    int num_boxes = static_cast<int>(output_shape[1]);
    int attributes = static_cast<int>(output_shape[2]);
  
    // Loop over detections
    for (int i = 0; i < num_boxes; i++) {
        float x1 = output_data[i * attributes + 0];
        float y1 = output_data[i * attributes + 1];
        float x2 = output_data[i * attributes + 2];
        float y2 = output_data[i * attributes + 3];
        float conf = output_data[i * attributes + 4];
        int cls_id = static_cast<int>(output_data[i * attributes + 5]);

        if (conf > 0.25f) { // lower threshold to see more detections
            int X1 = static_cast<int>(x1 * w / 640);
            int Y1 = static_cast<int>(y1 * h / 640);
            int X2 = static_cast<int>(x2 * w / 640);
            int Y2 = static_cast<int>(y2 * h / 640);

            cv::rectangle(img, cv::Point(X1, Y1), cv::Point(X2, Y2), cv::Scalar(0, 255, 0), 2);
            cv::putText(img, "ID:" + std::to_string(cls_id) + " " + std::to_string(conf),
                cv::Point(X1, Y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(0, 255, 0), 1);
        }
    }

    cv::imshow("YOLO26", img);
    cv::waitKey(0);
    return 0;
}

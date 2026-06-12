#include "scrfd.hpp"
#include <opencv2/opencv.hpp>
#include <cstring>
#include "rknpu2/scrfd_rknn.h"

SCRFD::SCRFD(const std::string& model_path) {
    memset(&rknn_ctx_, 0, sizeof(rknn_app_context_t));
    int ret = init_scrfd_model(model_path.c_str(), &rknn_ctx_);
    if (ret != 0) {
        throw std::runtime_error("init_scrfd_model failed");
    }
}

SCRFD::~SCRFD() {
    release_scrfd_model(&rknn_ctx_);
}

void SCRFD::detect(const cv::Mat& img, std::vector<FaceDetection>& out_dets,
                   float thresh, const cv::Size& input_size) {
    out_dets.clear();
    // Convert cv::Mat (BGR) to image_buffer_t (RGB888)
    image_buffer_t src_img;
    memset(&src_img, 0, sizeof(src_img));
    src_img.width = img.cols;
    src_img.height = img.rows;
    src_img.format = IMAGE_FORMAT_RGB888;
    src_img.width_stride = img.cols;
    src_img.height_stride = img.rows;
    src_img.size = src_img.width * src_img.height * 3;
    src_img.virt_addr = (unsigned char*)malloc(src_img.size);
    if (!src_img.virt_addr) return;
    cv::Mat rgb;
    cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
    if (rgb.isContinuous()) {
        memcpy(src_img.virt_addr, rgb.data, src_img.size);
    } else {
        for (int y = 0; y < rgb.rows; y++)
            memcpy(src_img.virt_addr + y * rgb.cols * 3, rgb.ptr(y), rgb.cols * 3);
    }

    object_detect_result_list od_results;
    memset(&od_results, 0, sizeof(od_results));
    int ret = inference_scrfd_model(&rknn_ctx_, &src_img, &od_results, thresh);
    if (ret == 0) {
        for (int i = 0; i < od_results.count; i++) {
            auto &r = od_results.results[i];
            FaceDetection d;
            d.x1 = r.box.left;
            d.y1 = r.box.top;
            d.x2 = r.box.right;
            d.y2 = r.box.bottom;
            d.score = r.prop;
            for (int k = 0; k < 5; k++) {
                d.kps[k][0] = r.kps[k][0];
                d.kps[k][1] = r.kps[k][1];
            }
            out_dets.push_back(d);
        }
    }
    free(src_img.virt_addr);
}

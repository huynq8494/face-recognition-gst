// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rknpu2/scrfd_rknn.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include <opencv2/opencv.hpp>
#include <chrono>


/*-------------------------------------------
              SCRFD Face Detection Demo
-------------------------------------------*/
int main(int argc, char **argv)
{
    // Usage:
    // 1) ./test09 <image_path>
    //    -> Uses default SCRFD model: /home/huynq/models/scrfd/scrfd_500m_bnkps_shape640x640.rknn
    // 2) ./test09 <image_path> <conf_threshold>
    //    -> Uses default SCRFD model with custom confidence (0.0-1.0)
    // 3) ./test09 <model_path> <image_path> [conf_threshold]
    //    -> Uses specified RKNN model
    const char *default_model_path = "/home/huynq/models/scrfd/scrfd_500m_bnkps_shape640x640.rknn";

    const char *model_path = NULL;
    const char *image_path = NULL;

    auto is_number = [](const char *s)->bool{
        if (!s || !*s) return false;
        char *endptr = NULL;
        strtod(s, &endptr);
        return endptr && *endptr == '\0';
    };

    float conf_thresh = 0.4f;

    if (argc == 2) {
        // only image provided -> use default model
        model_path = default_model_path;
        image_path = argv[1];
    } else if (argc == 3) {
        // could be: <model> <image> OR <image> <conf>
        if (is_number(argv[2])) {
            model_path = default_model_path;
            image_path = argv[1];
            conf_thresh = (float)atof(argv[2]);
        } else {
            model_path = argv[1];
            image_path = argv[2];
        }
    } else if (argc == 4) {
        model_path = argv[1];
        image_path = argv[2];
        if (is_number(argv[3])) conf_thresh = (float)atof(argv[3]);
    } else {
        printf("SCRFD Face Detection Demo\n");
        printf("Usage:\n");
        printf("  %s <image_path>\n", argv[0]);
        printf("  %s <image_path> <conf_threshold>\n", argv[0]);
        printf("  %s <model_path> <image_path> [conf_threshold]\n", argv[0]);
        printf("Default model: %s\n", default_model_path);
        return -1;
    }
    

    int ret;
    rknn_app_context_t rknn_app_ctx;
    image_buffer_t src_img;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    // Initialize RKNN model (uses hardware accelerator on supported boards)
    ret = init_scrfd_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_scrfd_model fail! ret=%d model_path=%s\n", ret, model_path);
        release_scrfd_model(&rknn_app_ctx);
        return -1;
    }

    // Open input video (mp4) and display realtime
    cv::VideoCapture cap(image_path);
    if (!cap.isOpened()) {
        printf("open video fail! path=%s\n", image_path);
        release_scrfd_model(&rknn_app_ctx);
        return -1;
    }

    const int AVG_FPS_COUNT = 100;
    int frame_idx = 0;
    double fps_sum = 0.0;
    double avg_fps = 0.0;

    cv::namedWindow("SCRFD", cv::WINDOW_AUTOSIZE);

    while (true) {
        cv::Mat frame_bgr;
        if (!cap.read(frame_bgr)) break; // end of video

        frame_idx++;

        // Convert cv::Mat (BGR) to image_buffer_t (RGB888)        
        memset(&src_img, 0, sizeof(src_img));
        src_img.width = frame_bgr.cols;
        src_img.height = frame_bgr.rows;
        src_img.format = IMAGE_FORMAT_RGB888;
        src_img.width_stride = frame_bgr.cols;
        src_img.height_stride = frame_bgr.rows;
        src_img.size = src_img.width * src_img.height * 3;
        src_img.virt_addr = (unsigned char*)malloc(src_img.size);
        if (!src_img.virt_addr) break;
        cv::Mat rgb;
        cv::cvtColor(frame_bgr, rgb, cv::COLOR_BGR2RGB);
        if (rgb.isContinuous()) {
            memcpy(src_img.virt_addr, rgb.data, src_img.size);
        } else {
            for (int y = 0; y < rgb.rows; y++)
                memcpy(src_img.virt_addr + y * rgb.cols * 3, rgb.ptr(y), rgb.cols * 3);
        }

        object_detect_result_list od_results;
        memset(&od_results, 0, sizeof(od_results));

        // Run inference and measure time
        auto t0 = std::chrono::steady_clock::now();
        ret = inference_scrfd_model(&rknn_app_ctx, &src_img, &od_results, conf_thresh);
        auto t1 = std::chrono::steady_clock::now();
        if (ret != 0)
        {
            printf("inference_scrfd_model fail! ret=%d\n", ret);
            free(src_img.virt_addr);
            break;
        }

        double elapsed_s = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        double cur_fps = elapsed_s > 0.0 ? (1.0 / elapsed_s) : 0.0;
        if (frame_idx <= AVG_FPS_COUNT) {
            fps_sum += cur_fps;
            if (frame_idx == AVG_FPS_COUNT) avg_fps = fps_sum / (double)AVG_FPS_COUNT;
        }

        // Draw boxes and confidence scores onto BGR frame
        char text[128];
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result *det_result = &(od_results.results[i]);
            int x1 = det_result->box.left;
            int y1 = det_result->box.top;
            int x2 = det_result->box.right;
            int y2 = det_result->box.bottom;
            cv::rectangle(frame_bgr, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(255,0,0), 2);
            snprintf(text, sizeof(text), "face %.1f%%", det_result->prop * 100.0f);
            int baseline = 0;
            cv::Size tsize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            int ty = std::max(0, y1 - tsize.height - 4);
            cv::putText(frame_bgr, text, cv::Point(x1, ty), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,0,255), 1);
        }

        // Overlay FPS
        if (avg_fps > 0.0) {
            snprintf(text, sizeof(text), "FPS: %.2f (avg@%d: %.2f)", cur_fps, AVG_FPS_COUNT, avg_fps);
        } else {
            snprintf(text, sizeof(text), "FPS: %.2f", cur_fps);
        }
        cv::putText(frame_bgr, text, cv::Point(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,255,0), 2);

        cv::imshow("SCRFD", frame_bgr);
        int key = cv::waitKey(1);
        if (key == 27) { // ESC
            free(src_img.virt_addr);
            break;
        }

        free(src_img.virt_addr);
    }

    cap.release();
    cv::destroyAllWindows();

    ret = release_scrfd_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_scrfd_model fail! ret=%d\n", ret);
    }

    if (src_img.virt_addr != NULL)
    {
        free(src_img.virt_addr);
    }

    return 0;
}

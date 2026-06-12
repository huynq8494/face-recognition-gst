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
              SCRFD Face Recognition Demo
           Integration of RKNN Detection +
            ONNX EdgeFace Recognition
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "scrfd_rknn.hpp"
#include "edgeface.hpp"
#include "face_recognition_system.hpp"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "gst_pipeline.hpp"

#include <opencv2/opencv.hpp>
#include <gst/gst.h>
#include <chrono>
#include <iostream>

int main(int argc, char **argv)
{
    // Usage:
    // 1) ./test12 <image_path>
    //    -> Detection + recognition on image with no gallery
    // 2) ./test12 <image_path> <conf_threshold>
    //    -> Detection + recognition on image with custom confidence
    // 3) ./test12 <video_path>
    //    -> Real-time face recognition on video (mp4/avi/mov/mkv)
    // 4) ./test12 <image_or_video> <conf_threshold> <gallery_path>
    //    -> With face gallery for recognition
    
    const char *default_scrfd_model = "/home/huynq/models/scrfd/scrfd_500m_bnkps_shape640x640.rknn";
    const char *default_edgeface_model = "/home/huynq/models/edgeface/edgeface_xs_gamma_06.onnx";
    
    const char *scrfd_model_path = NULL;
    const char *image_path = NULL;
    const char *gallery_path = NULL;
    float conf_thresh = 0.4f;

    auto is_number = [](const char *s)->bool{
        if (!s || !*s) return false;
        char *endptr = NULL;
        strtod(s, &endptr);
        return endptr && *endptr == '\0';
    };

    // Parse arguments
    if (argc == 2) {
        // only image provided -> use default models
        scrfd_model_path = default_scrfd_model;
        image_path = argv[1];
    } else if (argc == 3) {
        // could be: <model> <image> OR <image> <conf>
        if (is_number(argv[2])) {
            scrfd_model_path = default_scrfd_model;
            image_path = argv[1];
            conf_thresh = (float)atof(argv[2]);
        } else {
            scrfd_model_path = argv[1];
            image_path = argv[2];
        }
    } else if (argc >= 4) {
        if (is_number(argv[2])) {
            // <image> <conf> [gallery]
            scrfd_model_path = default_scrfd_model;
            image_path = argv[1];
            conf_thresh = (float)atof(argv[2]);
            if (argc >= 4) gallery_path = argv[3];
        } else {
            // <model> <image> <conf> [gallery]
            scrfd_model_path = argv[1];
            image_path = argv[2];
            if (is_number(argv[3])) conf_thresh = (float)atof(argv[3]);
            if (argc >= 5) gallery_path = argv[4];
        }
    } else {
        printf("SCRFD + EdgeFace Recognition Demo (Image & Video)\n");
        printf("Usage:\n");
        printf("  %s <image_path>                                    [image only]\n", argv[0]);
        printf("  %s <image_path> <conf_threshold>                   [image with confidence]\n", argv[0]);
        printf("  %s <image_path> <conf_threshold> <gallery_path>    [image with gallery]\n", argv[0]);
        printf("  %s <video_path>                                    [video: mp4/avi/mov/mkv]\n", argv[0]);
        printf("  %s <video_path> <conf_threshold>                   [video with confidence]\n", argv[0]);
        printf("  %s <video_path> <conf_threshold> <gallery_path>    [video with gallery]\n", argv[0]);
        printf("\nDefault SCRFD model: %s\n", default_scrfd_model);
        printf("Default EdgeFace model: %s\n", default_edgeface_model);
        printf("Default confidence threshold: %.2f\n", conf_thresh);
        return -1;
    }

    // Check if input is image or video/stream
    bool is_video = false;
    const char *ext = strrchr(image_path, '.');
    if (ext && (strcasecmp(ext, ".mp4") == 0 || strcasecmp(ext, ".avi") == 0 || 
                strcasecmp(ext, ".mov") == 0 || strcasecmp(ext, ".mkv") == 0)) {
        is_video = true;
    } else if (strncasecmp(image_path, "rtsp://", 7) == 0 ||
               strncmp(image_path, "/dev/video", 10) == 0) {
        is_video = true;
    }

    gst_init(&argc, &argv);

    try {
        // Initialize detectors
        printf("\n[Init] Loading SCRFD RKNN model...\n");
        SCRFD_RKNN detector(scrfd_model_path);
        
        printf("[Init] Loading EdgeFace ONNX model...\n");
        EdgeFace recognizer(default_edgeface_model);
        
        // Create recognition system
        printf("[Init] Creating FaceRecognitionSystem...\n");
        FaceRecognitionSystem system(
            detector, recognizer,
            /*rec_threshold=*/  0.5f,
            /*re_id_interval=*/ 30,
            /*iou_thresh=*/     0.4f,
            /*max_lost=*/       10,
            /*min_hits=*/       3);
        
        // Load gallery if provided
        if (gallery_path) {
            printf("[Init] Loading gallery from: %s\n", gallery_path);
            try {
                system.loadGallery(gallery_path);
            } catch (const std::exception& e) {
                printf("[Warning] Could not load gallery: %s\n", e.what());
            }
        }

        if (!is_video) {
            // Process image
            printf("\n[Processing] Image: %s\n", image_path);
            
            // Check if file exists first
            FILE* test_file = fopen(image_path, "rb");
            if (!test_file) {
                printf("Error: Cannot open file (permission/not found): %s\n", image_path);
                return -1;
            }
            fclose(test_file);
            
            cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
            if (image.empty()) {
                printf("Error: Failed to decode image with cv::imread(): %s\n", image_path);
                printf("Note: OpenCV may be missing JPEG/PNG codec support\n");
                printf("Try installing: sudo apt-get install libopencv-dev libjasper1 libtiff5\n");
                return -1;
            }
            printf("Image loaded: %d x %d\n", image.cols, image.rows);
            
            // Process frame
            auto start = std::chrono::high_resolution_clock::now();
            std::vector<FaceResult> results = system.processFrame(image, conf_thresh, cv::Size(640, 640));
            auto end = std::chrono::high_resolution_clock::now();
            float inference_time = std::chrono::duration<float>(end - start).count() * 1000.0f;
            
            printf("\n[Result] Detected %zu faces\n", results.size());
            printf("[Performance] Inference time: %.2f ms\n", inference_time);
            
            // Print results
            for (const auto& res : results) {
                printf("  Track ID %d: bbox=(%.1f,%.1f,%.1f,%.1f) label=%s score=%.3f\n",
                       res.track_id, res.bbox[0], res.bbox[1], res.bbox[2], res.bbox[3],
                       res.label.empty() ? "Unknown" : res.label.c_str(),
                       res.rec_score);
            }
            
            // Draw visualization
            cv::Mat vis = FaceRecognitionSystem::draw(image, results, 1000.0f/inference_time, 0.f);
            
            // Save output
            std::string output_path = "output_recognized.png";
            cv::imwrite(output_path, vis);
            printf("\n[Output] Saved to: %s\n", output_path);
            
            // Display (if possible)
            cv::imshow("Face Recognition Results", vis);
            printf("[Display] Press any key to exit...\n");
            cv::waitKey(0);
            cv::destroyAllWindows();

        } else {
            // Process video / stream via GStreamer pipeline
            printf("\n[Processing] Video/stream: %s\n", image_path);

            int frame_count = 0;
            int total_faces = 0;
            double total_time = 0.0;
            FPSCounter fps_counter(50, 100);
            std::atomic<bool> quit{false};

            cv::namedWindow("Face Recognition - Video", cv::WINDOW_AUTOSIZE);

            auto on_frame = [&](const cv::Mat& frame) {
                if (quit) return;

                frame_count++;
                auto start = std::chrono::high_resolution_clock::now();
                std::vector<FaceResult> results = system.processFrame(frame, conf_thresh, cv::Size(640, 640));
                auto end = std::chrono::high_resolution_clock::now();
                float inference_time = std::chrono::duration<float>(end - start).count() * 1000.0f;

                total_time += inference_time;
                total_faces += results.size();
                fps_counter.tick();

                // Draw visualization
                cv::Mat vis = FaceRecognitionSystem::draw(frame, results, fps_counter.fps, fps_counter.avg_fps);

                if (frame_count % 10 == 0) {
                    printf("[Frame %4d] Faces: %zu | FPS: %.1f | Avg: %.1f\n",
                           frame_count, results.size(), fps_counter.fps, fps_counter.avg_fps);
                }

                cv::imshow("Face Recognition - Video", vis);
                int key = cv::waitKey(1);
                if (key == 27 || key == 'q') {
                    printf("\nStopped by user\n");
                    quit = true;
                }
            };

            FaceGstPipeline pipeline(image_path, on_frame);
            std::thread gst_thread([&]{ pipeline.run(); });

            while (!quit) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }

            pipeline.stop();
            gst_thread.join();
            cv::destroyAllWindows();

            printf("\n[Video Summary]\n");
            printf("  Frames processed: %d\n", frame_count);
            printf("  Total faces detected: %d\n", total_faces);
            printf("  Average faces per frame: %.2f\n", frame_count > 0 ? (float)total_faces / frame_count : 0.0f);
            printf("  Total processing time: %.2f s\n", total_time / 1000.0f);
            printf("  Average FPS: %.2f\n", frame_count > 0 ? total_time > 0 ? (frame_count * 1000.0f) / total_time : 0.0f : 0.0f);
        }
        
    } catch (const std::exception& e) {
        printf("[Error] %s\n", e.what());
        return -1;
    }
    
    return 0;
}

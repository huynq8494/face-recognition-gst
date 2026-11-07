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
#include "gst_streamer.hpp"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

#include <opencv2/opencv.hpp>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

class GstInputSource {
public:
    GstInputSource(const std::string& uri, bool is_video);
    ~GstInputSource();

    bool read(cv::Mat& frame);
    bool isOpened() const { return opened_; }
    float fps() const { return fps_; }
    cv::Size frameSize() const { return frame_size_; }

private:
    std::string buildPipeline(const std::string& uri, bool is_video);
    bool init(const std::string& uri, bool is_video);

    GstElement* pipeline_ = nullptr;
    GstElement* appsink_ = nullptr;
    bool opened_ = false;
    cv::Size frame_size_ = cv::Size(0, 0);
    float fps_ = 25.0f;
};

static std::string gstQuoteUri(const std::string& path) {
    GError* err = nullptr;
    gchar* uri = g_filename_to_uri(path.c_str(), nullptr, &err);
    if (!uri) {
        std::string msg = err ? err->message : "unknown";
        if (err) g_error_free(err);
        throw std::runtime_error("Failed to convert filename to URI: " + msg);
    }
    std::string result(uri);
    g_free(uri);
    return result;
}

GstInputSource::GstInputSource(const std::string& uri, bool is_video) {
    gst_init(nullptr, nullptr);
    if (!init(uri, is_video)) {
        throw std::runtime_error("Failed to initialize GStreamer input source");
    }
}

GstInputSource::~GstInputSource() {
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
}

bool GstInputSource::init(const std::string& uri, bool is_video) {
    std::string pipeline_desc = buildPipeline(uri, is_video);
    GError* err = nullptr;
    pipeline_ = gst_parse_launch(pipeline_desc.c_str(), &err);
    if (!pipeline_) {
        std::string msg = err ? err->message : "unknown pipeline error";
        if (err) g_error_free(err);
        return false;
    }

    appsink_ = gst_bin_get_by_name(GST_BIN(pipeline_), "sink");
    if (!appsink_) {
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        return false;
    }

    g_object_set(appsink_,
                 "emit-signals", false,
                 "sync", false,
                 "max-buffers", 1,
                 "drop", true,
                 nullptr);

    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    opened_ = true;
    return true;
}

std::string GstInputSource::buildPipeline(const std::string& uri, bool is_video) {
    if (uri.rfind("/dev/video", 0) == 0) {
        return "v4l2src device=" + uri +
               " ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink";
    }

    std::string uri_quoted = gstQuoteUri(uri);
    return "uridecodebin uri=\"" + uri_quoted +
           "\" ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink";
}

bool GstInputSource::read(cv::Mat& frame) {
    if (!opened_ || !appsink_)
        return false;

    GstSample* sample = gst_app_sink_pull_sample(GST_APP_SINK(appsink_));
    if (!sample)
        return false;

    GstCaps* caps = gst_sample_get_caps(sample);
    if (!caps) {
        gst_sample_unref(sample);
        return false;
    }

    GstVideoInfo info;
    if (!gst_video_info_from_caps(&info, caps)) {
        gst_sample_unref(sample);
        return false;
    }

    GstBuffer* buffer = gst_sample_get_buffer(sample);
    if (!buffer) {
        gst_sample_unref(sample);
        return false;
    }

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return false;
    }

    if (info.finfo->format != GST_VIDEO_FORMAT_BGR) {
        gst_buffer_unmap(buffer, &map);
        gst_sample_unref(sample);
        return false;
    }

    cv::Mat mat(info.height, info.width, CV_8UC3, map.data, info.stride[0]);
    frame = mat.clone();

    if (frame_size_.width == 0 || frame_size_.height == 0) {
        frame_size_ = cv::Size(info.width, info.height);
    }
    if (info.fps_d != 0 && info.fps_n != 0) {
        fps_ = static_cast<float>(info.fps_n) / static_cast<float>(info.fps_d);
    }

    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    return true;
}

int main(int argc, char **argv)
{
    // Usage:
    // 1) ./test12 <image_path>
    // 2) ./test12 <video_path> 
    
    const char *scrfd_model_path = "/home/hw/models/scrfd/scrfd_500m_bnkps_shape640x640.rknn";
    const char *edgeface_model_path = "/home/hw/models/edgeface/edgeface_xs_gamma_06.onnx";
    const char *gallery_path = "/home/hw/data/embeddings/face_db_530.bin";
    float conf_thresh = 0.4f;
    std::string rtp_host = "10.0.0.1";
    int rtp_port = 5000;
    int output_fps = 25;

    if (argc < 2) {
        printf("SCRFD + EdgeFace Recognition Demo (Image & Video)\n");
        printf("Usage:\n");
        printf("  %s <input_path>                                    [image only]\n", argv[0]);
        printf("  %s <video_path>                                    [video: mp4/avi/mov/mkv or /dev/video*]\n", argv[0]);
        return 0;
    }

    const char *input_path = argv[1];

    for (int i = 0; i < argc; i ++) {
        printf("argv[%d]=%s\n", i, argv[i]);
    }

    bool use_rtp = !rtp_host.empty() && rtp_port > 0;

    // Check if input is image or video
    bool is_video = false;
    const char *ext = strrchr(input_path, '.');
    if (ext && (strcasecmp(ext, ".mp4") == 0 || strcasecmp(ext, ".avi") == 0 || 
                strcasecmp(ext, ".mov") == 0 || strcasecmp(ext, ".mkv") == 0)) {
        is_video = true;
    }
    if (strncmp(input_path, "/dev/video", 10) == 0) {
        is_video = true;
    }

    try {
        // Initialize detectors
        printf("\n[Init] Loading SCRFD RKNN model %s:...\n", scrfd_model_path);
        SCRFD_RKNN detector(scrfd_model_path);
        
        printf("[Init] Loading EdgeFace ONNX model...\n");
        
        // Create recognition system
        EdgeFace recognizer(edgeface_model_path);
        printf("[Init] Creating FaceRecognitionSystem...\n");
        FaceRecognitionSystem system(
            detector, recognizer,
            /*rec_threshold=*/  0.5f,
            /*re_id_interval=*/ 30,
            /*iou_thresh=*/     0.4f,
            /*max_lost=*/       10,
            /*min_hits=*/       3);

        std::unique_ptr<GstRtpStreamer> streamer;
        bool show_window = !use_rtp;
        
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
            printf("\n[Processing] Image: %s\n", input_path);

            GstInputSource source(input_path, false);
            if (!source.isOpened()) {
                printf("Error: Failed to open image input with GStreamer: %s\n", input_path);
                return -1;
            }

            cv::Mat image;
            if (!source.read(image) || image.empty()) {
                printf("Error: Failed to read image frame via GStreamer: %s\n", input_path);
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

            if (use_rtp) {
                streamer = std::make_unique<GstRtpStreamer>(rtp_host, rtp_port,
                    vis.cols, vis.rows, output_fps);
                streamer->pushFrame(vis);
                printf("[RTP] Sent image frame to %s:%d\n", rtp_host.c_str(), rtp_port);
            } else {
                // Display (if possible)
                cv::imshow("Face Recognition Results", vis);
                printf("[Display] Press any key to exit...\n");
                cv::waitKey(0);
                cv::destroyAllWindows();
            }

        } else {
            // Process video
            printf("\n[Processing] Video: %s\n", input_path);
            GstInputSource source(input_path, true);
            if (!source.isOpened()) {
                printf("Failed to open video input with GStreamer: %s\n", input_path);
                return -1;
            }

            int frame_count = 0;
            int total_faces = 0;
            double total_time = 0.0;
            FPSCounter fps_counter(50, 100);

            if (show_window) {
                cv::namedWindow("Face Recognition - Video", cv::WINDOW_AUTOSIZE);
            }

            cv::Mat frame;
            if (!source.read(frame) || frame.empty()) {
                printf("Failed to read first video frame: %s\n", input_path);
                return -1;
            }

            double input_fps = source.fps();
            if (input_fps > 1.0)
                output_fps = (int)input_fps;

            printf("Video properties:\n");
            printf("  FPS: %.2f\n", input_fps);
            printf("  Width: %d, Height: %d\n", source.frameSize().width, source.frameSize().height);
            printf("  Total frames: unknown\n");
            printf("\nProcessing%s...\n", show_window ? " (press ESC to stop)" : "");

            do {
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
                
                if (use_rtp && !streamer) {
                    streamer = std::make_unique<GstRtpStreamer>(rtp_host, rtp_port,
                        vis.cols, vis.rows, output_fps);
                    printf("[RTP] Streaming to %s:%d\n", rtp_host.c_str(), rtp_port);
                }

                if (use_rtp) {
                    streamer->pushFrame(vis);
                }

                // Print frame info every 10 frames
                if (frame_count % 10 == 0) {
                    printf("[Frame %4d] Faces: %zu | FPS: %.1f | Avg: %.1f\n",
                           frame_count, results.size(), fps_counter.fps, fps_counter.avg_fps);
                }
                
                if (show_window) {
                    cv::imshow("Face Recognition - Video", vis);
                    int key = cv::waitKey(1);
                    if (key == 27) { // ESC
                        printf("\nStopped by user\n");
                        break;
                    }
                }
            } while (source.read(frame));

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

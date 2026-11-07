#include <gst/gst.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <atomic>
#include <memory>
#include "scrfd.hpp"
#include "edgeface.hpp"
#include "face_recognition_system.hpp"
#include "gst_pipeline.hpp"
#include <thread>

int main(int argc, char* argv[]) {
    gst_init(&argc, &argv);

    // ── Models ────────────────────────────────────────────────
    const std::string scrfd_model = "/home/hw/models/scrfd/scrfd_500m_bnkps_shape640x640.rknn";
    const std::string edgeface_model = "/home/hw/models/edgeface/edgeface_xs_gamma_06.onnx";

    std::unique_ptr<SCRFD> detector;
    std::unique_ptr<EdgeFace> recognizer;

    try {
        detector = std::make_unique<SCRFD>(scrfd_model);
        recognizer = std::make_unique<EdgeFace>(edgeface_model);
    } catch (const std::exception& e) {
        std::cerr << "Model initialization failed: " << e.what() << "\n";
        return 1;
    }

    // ── Recognition system ────────────────────────────────────
    FaceRecognitionSystem system(
        *detector, *recognizer,
        /*rec_threshold=*/  0.5f,
        /*re_id_interval=*/ 30,
        /*iou_thresh=*/     0.4f,
        /*max_lost=*/       10,
        /*min_hits=*/       3);

    system.loadGallery("/home/hw/data/embeddings/face_db.bin");

    // ── FPS ───────────────────────────────────────────────────
    FPSCounter fps_counter(/*warmup=*/100, /*window=*/200);

    std::atomic<bool> quit{false};

    // ── Frame callback (runs on GStreamer thread) ──────────────
    auto on_frame = [&](const cv::Mat& frame) {
        if (quit) return;

        auto results = system.processFrame(frame, 0.5f, cv::Size(640, 640));
        fps_counter.tick();

        auto vis = FaceRecognitionSystem::draw(
            frame, results, fps_counter.fps, fps_counter.avg_fps);

        cv::imshow("Face Recognition", vis);
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {   // q or ESC
            quit = true;
        }
    };

    // ── Pipeline ──────────────────────────────────────────────
    std::string uri = "/home/hw/data/videos/test_domain.mp4";
    if (argc > 1) uri = argv[1];   // optional override from CLI

    try {
        FaceGstPipeline pipeline(uri, on_frame);

        // run() blocks; quit flag stops it from inside the callback
        // We also need a way to break the run() loop from the callback.
        // Simplest: run in a thread, join after quit flag is set.
        std::thread gst_thread([&]{ pipeline.run(); });

        // Wait for quit signal (set by callback or pipeline EOS)
        while (!quit) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            // Also stop if pipeline finished naturally
        }
        pipeline.stop();
        gst_thread.join();

    } catch (const std::exception& e) {
        std::cerr << "Fatal: " << e.what() << "\n";
        return 1;
    }

    cv::destroyAllWindows();
    return 0;
}

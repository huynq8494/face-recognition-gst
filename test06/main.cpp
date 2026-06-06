#include <opencv2/opencv.hpp>
#include <iostream>
#include "scrfd.hpp"
#include "edgeface.hpp"
#include "face_recognition_system.hpp"

int main() {
    // ── Models ────────────────────────────────────────────────
    SCRFD detector(
        "/home/huynq/models/scrfd/scrfd_500m_bnkps_shape640x640.onnx");

    EdgeFace recognizer(
        "/home/huynq/models/edgeface/edgeface_xs_gamma_06.onnx");

    // ── System ────────────────────────────────────────────────
    FaceRecognitionSystem system(
        detector, recognizer,
        /*rec_threshold=*/  0.5f,
        /*re_id_interval=*/ 30,
        /*iou_thresh=*/     0.4f,
        /*max_lost=*/       10,
        /*min_hits=*/       3);

    system.loadGallery("face_db.bin");

    // ── Video ─────────────────────────────────────────────────
    cv::VideoCapture cap("/home/huynq/data/videos/test_domain.mp4");
    if (!cap.isOpened()) { std::cerr << "Failed to open video\n"; return -1; }

    FPSCounter fps_counter(/*warmup=*/100, /*window=*/200);
    cv::Mat frame;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        auto results = system.processFrame(frame, 0.5f, cv::Size(640, 640));

        fps_counter.tick();

        std::cout << "[Frame " << system.frameIdx() << "]"
                  << "  faces="   << results.size()
                  << "  FPS="     << fps_counter.fps
                  << "  AvgFPS="  << fps_counter.avg_fps
                  << (fps_counter.isWarmedUp() ? "" : "  [warming up]")
                  << "\n";

        auto vis = FaceRecognitionSystem::draw(
            frame, results, fps_counter.fps, fps_counter.avg_fps);

        cv::imshow("FaceRecognition", vis);
        if (cv::waitKey(1) == 'q') break;
    }

    std::cout << "\n─────────────────────────────────────────────\n"
              << "  Total frames : " << system.frameIdx()      << "\n"
              << "  Average FPS  : " << fps_counter.avg_fps    << "\n"
              << "─────────────────────────────────────────────\n";

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

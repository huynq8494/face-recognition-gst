#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>
#include <array>
#include "scrfd_rknn.hpp"
#include "edgeface.hpp"
#include "tracker.hpp"

// ─────────────────────────────────────────────
//  Binary DB loader
// ─────────────────────────────────────────────
std::map<std::string, std::vector<std::vector<float>>>
loadFaceDb(const std::string& db_path);

// ─────────────────────────────────────────────
//  Per-frame output
// ─────────────────────────────────────────────
struct FaceResult {
    int                  track_id;
    std::array<float, 5> bbox;       // x1,y1,x2,y2,score
    std::vector<float>   embedding;
    std::string          label;
    float                rec_score = 0.f;
    bool                 is_new    = false;
};

// ─────────────────────────────────────────────
//  FPS Counter
// ─────────────────────────────────────────────
class FPSCounter {
public:
    FPSCounter(int warmup_frames = 100, int avg_window = 200);
    void  tick();
    bool  isWarmedUp() const { return frame_ > warmup_frames_; }
    float fps     = 0.f;
    float avg_fps = 0.f;
    int   frame_  = 0;

private:
    int    warmup_frames_;
    int    avg_window_;
    double t_last_ = -1.0;
    std::vector<float> fps_history_;
};

// ─────────────────────────────────────────────
//  Face Recognition System
// ─────────────────────────────────────────────
class FaceRecognitionSystem {
public:
    FaceRecognitionSystem(
        SCRFD_RKNN& detector,
        EdgeFace&   recognizer,
        float rec_threshold  = 0.5f,
        int   re_id_interval = 30,
        float iou_thresh     = 0.4f,
        int   max_lost       = 10,
        int   min_hits       = 3);

    void loadGallery(const std::string& db_path);
    void registerFace(const std::string& label, const cv::Mat& img,
                      const FaceDetection* det = nullptr);

    std::vector<FaceResult> processFrame(const cv::Mat& img,
                                          float det_thresh  = 0.5f,
                                          cv::Size input_size = cv::Size(640, 640));

    static cv::Mat draw(const cv::Mat& img, const std::vector<FaceResult>& results,
                        float fps = 0.f, float avg_fps = 0.f);

    int frameIdx() const { return frame_idx_; }

private:
    SCRFD_RKNN& detector_;
    EdgeFace&   recognizer_;
    float       rec_threshold_;
    int         re_id_interval_;
    IoUTracker  tracker_;
    int         frame_idx_ = 0;

    // gallery: name → list of L2-normalised embeddings
    std::map<std::string, std::vector<std::vector<float>>> gallery_;

    struct TrackCache {
        std::vector<float> embedding;
        std::string        label;
        float              rec_score = 0.f;
        int                frame     = 0;
    };
    std::map<int, TrackCache> track_cache_;

    std::pair<std::string, float> matchGallery(const std::vector<float>& emb);
};

#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <string>
#include <array>

// ─────────────────────────────────────────────
//  Kalman Filter per track
//  State  : [cx, cy, w, h, vcx, vcy, vw, vh]
//  Measure: [cx, cy, w, h]
// ─────────────────────────────────────────────
class KalmanBoxTracker {
public:
    explicit KalmanBoxTracker(const std::array<float, 5>& bbox);

    std::array<float, 5> predict();
    void update(const std::array<float, 5>& bbox);

    float last_score = 1.0f;

private:
    cv::KalmanFilter kf_;

    static std::array<float, 4> toZ(const std::array<float, 5>& bbox);
    static std::array<float, 5> toBBox(const cv::Mat& state, float score);
};

// ─────────────────────────────────────────────
//  Face Track
// ─────────────────────────────────────────────
struct FaceTrack {
    int                   track_id;
    KalmanBoxTracker      kalman;
    std::array<float, 5>  bbox;
    std::array<float, 5>  pred_bbox;
    std::vector<float>    embedding;
    int                   hits          = 1;
    int                   no_loss_count = 0;
    int                   age           = 0;
    std::string           label;

    FaceTrack(int id, const std::array<float, 5>& b)
        : track_id(id), kalman(b), bbox(b), pred_bbox(b) {}
};

// ─────────────────────────────────────────────
//  IoU + Kalman Tracker
// ─────────────────────────────────────────────
class IoUTracker {
public:
    struct TrackResult {
        int                  track_id;
        std::array<float, 5> bbox;
        bool                 is_new;
    };

    IoUTracker(float iou_thresh = 0.3f, int max_lost = 10, int min_hits = 3);

    std::vector<TrackResult> update(const std::vector<std::array<float, 5>>& bboxes);

    static float iou(const std::array<float, 5>& a, const std::array<float, 5>& b);

    std::map<int, FaceTrack> tracks;

private:
    float iou_thresh_;
    int   max_lost_;
    int   min_hits_;
    int   next_id_ = 1;

    static float centerDistance(const std::array<float, 5>& a,
                                 const std::array<float, 5>& b);
    float matchScore(const std::array<float, 5>& det,
                     const std::array<float, 5>& pred);
};

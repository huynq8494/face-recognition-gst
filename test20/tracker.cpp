#include "tracker.hpp"
#include <numeric>
#include <algorithm>
#include <cmath>
#include <set>

// ── KalmanBoxTracker ──────────────────────────────────────────
KalmanBoxTracker::KalmanBoxTracker(const std::array<float, 5>& bbox)
    : kf_(8, 4)
{
    // State transition (constant-velocity model)
    kf_.transitionMatrix = (cv::Mat_<float>(8, 8) <<
        1,0,0,0, 1,0,0,0,
        0,1,0,0, 0,1,0,0,
        0,0,1,0, 0,0,1,0,
        0,0,0,1, 0,0,0,1,
        0,0,0,0, 1,0,0,0,
        0,0,0,0, 0,1,0,0,
        0,0,0,0, 0,0,1,0,
        0,0,0,0, 0,0,0,1);

    // Measurement matrix (observe cx, cy, w, h only)
    kf_.measurementMatrix = (cv::Mat_<float>(4, 8) <<
        1,0,0,0, 0,0,0,0,
        0,1,0,0, 0,0,0,0,
        0,0,1,0, 0,0,0,0,
        0,0,0,1, 0,0,0,0);

    // Process noise: velocity terms get higher noise
    cv::setIdentity(kf_.processNoiseCov, cv::Scalar(0.01f));
    for (int i = 4; i < 8; i++)
        kf_.processNoiseCov.at<float>(i, i) *= 5.0f;

    // Measurement noise
    cv::setIdentity(kf_.measurementNoiseCov, cv::Scalar(0.5f));

    // Initial error covariance (high uncertainty in velocity)
    cv::setIdentity(kf_.errorCovPost, cv::Scalar(10.0f));
    for (int i = 4; i < 8; i++)
        kf_.errorCovPost.at<float>(i, i) *= 100.0f;

    auto z = toZ(bbox);
    kf_.statePost = (cv::Mat_<float>(8, 1) << z[0], z[1], z[2], z[3], 0, 0, 0, 0);
    last_score = bbox[4];
}

std::array<float, 4> KalmanBoxTracker::toZ(const std::array<float, 5>& bbox) {
    float cx = (bbox[0] + bbox[2]) / 2;
    float cy = (bbox[1] + bbox[3]) / 2;
    float w  =  bbox[2] - bbox[0];
    float h  =  bbox[3] - bbox[1];
    return {cx, cy, w, h};
}

std::array<float, 5> KalmanBoxTracker::toBBox(const cv::Mat& state, float score) {
    float cx = state.at<float>(0), cy = state.at<float>(1);
    float w  = state.at<float>(2), h  = state.at<float>(3);
    return {cx - w/2, cy - h/2, cx + w/2, cy + h/2, score};
}

std::array<float, 5> KalmanBoxTracker::predict() {
    cv::Mat state = kf_.predict();
    return toBBox(state, last_score);
}

void KalmanBoxTracker::update(const std::array<float, 5>& bbox) {
    auto z = toZ(bbox);
    cv::Mat meas = (cv::Mat_<float>(4, 1) << z[0], z[1], z[2], z[3]);
    kf_.correct(meas);
    last_score = bbox[4];
}

// ── IoUTracker ────────────────────────────────────────────────
IoUTracker::IoUTracker(float iou_thresh, int max_lost, int min_hits)
    : iou_thresh_(iou_thresh), max_lost_(max_lost), min_hits_(min_hits) {}

float IoUTracker::iou(const std::array<float, 5>& a, const std::array<float, 5>& b) {
    float xA = std::max(a[0], b[0]), yA = std::max(a[1], b[1]);
    float xB = std::min(a[2], b[2]), yB = std::min(a[3], b[3]);
    float inter = std::max(0.f, xB - xA) * std::max(0.f, yB - yA);
    if (inter == 0.f) return 0.f;
    float aA = (a[2]-a[0]) * (a[3]-a[1]);
    float aB = (b[2]-b[0]) * (b[3]-b[1]);
    return inter / (aA + aB - inter);
}

float IoUTracker::centerDistance(const std::array<float, 5>& a,
                                  const std::array<float, 5>& b) {
    float cxA = (a[0]+a[2])/2, cyA = (a[1]+a[3])/2;
    float cxB = (b[0]+b[2])/2, cyB = (b[1]+b[3])/2;
    float wA  =  a[2]-a[0],    hA  =  a[3]-a[1];
    return std::sqrt((cxA-cxB)*(cxA-cxB) + (cyA-cyB)*(cyA-cyB))
           / (std::sqrt(wA * hA) + 1e-6f);
}

float IoUTracker::matchScore(const std::array<float, 5>& det,
                              const std::array<float, 5>& pred) {
    float iou_val    = iou(det, pred);
    float dist_score = std::max(0.f, 1.f - centerDistance(det, pred));
    return 0.7f * iou_val + 0.3f * dist_score;
}

std::vector<IoUTracker::TrackResult>
IoUTracker::update(const std::vector<std::array<float, 5>>& bboxes) {

    // 1. Kalman predict every track
    for (auto& [tid, track] : tracks) {
        track.pred_bbox = track.kalman.predict();
        track.age++;
    }

    std::vector<int> unmatched(bboxes.size());
    std::iota(unmatched.begin(), unmatched.end(), 0);
    std::set<int> matched_tracks;
    // map track id -> detection index for this update
    std::map<int,int> track_to_det;

    // 2. Greedy matching on combined score
    if (!tracks.empty() && !bboxes.empty()) {
        std::vector<int> track_ids;
        for (auto& [tid, _] : tracks) track_ids.push_back(tid);

        int nd = (int)bboxes.size(), nt = (int)track_ids.size();
        std::vector<float> score_mat(nd * nt);
        for (int di = 0; di < nd; di++)
            for (int ti = 0; ti < nt; ti++)
                score_mat[di*nt + ti] = matchScore(
                    bboxes[di], tracks.at(track_ids[ti]).pred_bbox);

        while (true) {
            auto it  = std::max_element(score_mat.begin(), score_mat.end());
            if (*it < iou_thresh_) break;

            int idx = (int)(it - score_mat.begin());
            int di  = idx / nt, ti = idx % nt;
            int tid = track_ids[ti];

            auto& track         = tracks.at(tid);
            track.kalman.update(bboxes[di]);
            track.bbox          = bboxes[di];
            track.hits++;
            track.no_loss_count = 0;
            matched_tracks.insert(tid);
            track_to_det[tid] = di;

            // Remove di from unmatched list
            auto uit = std::find(unmatched.begin(), unmatched.end(), di);
            if (uit != unmatched.end()) unmatched.erase(uit);

            // Suppress this detection and track from future matching
            for (int t = 0; t < nt; t++) score_mat[di*nt + t] = -1.f;
            for (int d = 0; d < nd; d++) score_mat[d*nt  + ti] = -1.f;
        }
    }

    // 3. Create new tracks for unmatched detections
    std::set<int> new_ids;
    for (int di : unmatched) {
        int tid = next_id_++;
        tracks.emplace(tid, FaceTrack(tid, bboxes[di]));
        matched_tracks.insert(tid);
        new_ids.insert(tid);
        track_to_det[tid] = di;
    }

    // 4. Age unmatched tracks, remove stale ones
    for (auto it = tracks.begin(); it != tracks.end(); ) {
        int tid = it->first;
        if (matched_tracks.find(tid) == matched_tracks.end()) {
            it->second.no_loss_count++;
            if (it->second.no_loss_count > max_lost_) {
                it = tracks.erase(it);
                continue;
            }
        }
        ++it;
    }

    // 5. Return confirmed tracks only (hits >= min_hits)
    std::vector<TrackResult> results;
    for (auto& [tid, track] : tracks) {
        if (track.hits < min_hits_) continue;

        // Use Kalman predicted bbox for smooth output, keep original score
        auto out_bbox    = track.pred_bbox;
        out_bbox[4]      = track.bbox[4];
        bool is_new      = (new_ids.count(tid) > 0) && (track.hits == 1);
        int det_idx = -1;
        auto it = track_to_det.find(tid);
        if (it != track_to_det.end()) det_idx = it->second;
        results.push_back({tid, out_bbox, is_new, det_idx});
    }
    return results;
}

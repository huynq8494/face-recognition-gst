#include "face_recognition_system.hpp"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>

// ─────────────────────────────────────────────
//  Binary DB loader
// ─────────────────────────────────────────────
static const char FACEDB_MAGIC[8] = {'F','A','C','E','D','B','\0','\0'};

std::map<std::string, std::vector<std::vector<float>>>
loadFaceDb(const std::string& db_path) {
    std::map<std::string, std::vector<std::vector<float>>> gallery;
    std::ifstream f(db_path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + db_path);

    char magic[8];
    f.read(magic, 8);
    if (std::memcmp(magic, FACEDB_MAGIC, 8) != 0)
        throw std::runtime_error("Not a valid FACEDB file: " + db_path);

    auto read_u32 = [&]() {
        uint32_t v; f.read(reinterpret_cast<char*>(&v), 4); return v;
    };

    uint32_t version = read_u32();
    if (version != 1)
        throw std::runtime_error("Unsupported FACEDB version: " + std::to_string(version));

    uint32_t num_ids = read_u32();
    uint32_t emb_dim = read_u32();

    for (uint32_t i = 0; i < num_ids; i++) {
        uint32_t name_len = read_u32();
        std::string name(name_len, '\0');
        f.read(name.data(), name_len);

        uint32_t num_embs = read_u32();
        std::vector<float> raw(num_embs * emb_dim);
        f.read(reinterpret_cast<char*>(raw.data()), num_embs * emb_dim * 4);

        for (uint32_t e = 0; e < num_embs; e++)
            gallery[name].emplace_back(raw.begin() + e*emb_dim,
                                       raw.begin() + (e+1)*emb_dim);
    }

    std::cout << "[DB] Loaded " << gallery.size() << " identities"
              << " (dim=" << emb_dim << ") from " << db_path << "\n";
    return gallery;
}

// ─────────────────────────────────────────────
//  FPSCounter
// ─────────────────────────────────────────────
FPSCounter::FPSCounter(int warmup_frames, int avg_window)
    : warmup_frames_(warmup_frames), avg_window_(avg_window) {}

void FPSCounter::tick() {
    using clock = std::chrono::steady_clock;
    static clock::time_point t_last;
    auto now = clock::now();
    frame_++;

    if (t_last_ >= 0.0) {
        double dt = std::chrono::duration<double>(now - t_last).count();
        fps = (float)(1.0 / std::max(dt, 1e-9));

        if (frame_ > warmup_frames_) {
            fps_history_.push_back(fps);
            if ((int)fps_history_.size() > avg_window_)
                fps_history_.erase(fps_history_.begin());
            float sum = 0.f;
            for (float v : fps_history_) sum += v;
            avg_fps = sum / fps_history_.size();
        }
    }
    t_last  = now;
    t_last_ = 0.0; // mark as initialized
}

// ─────────────────────────────────────────────
//  FaceRecognitionSystem
// ─────────────────────────────────────────────
FaceRecognitionSystem::FaceRecognitionSystem(
    SCRFD& detector, EdgeFace& recognizer,
    float rec_threshold, int re_id_interval,
    float iou_thresh, int max_lost, int min_hits)
    : detector_(detector),
      recognizer_(recognizer),
      rec_threshold_(rec_threshold),
      re_id_interval_(re_id_interval),
      tracker_(iou_thresh, max_lost, min_hits) {}

void FaceRecognitionSystem::loadGallery(const std::string& db_path) {
    auto loaded = loadFaceDb(db_path);
    for (auto& [name, embs] : loaded) {
        for (auto& e : embs)
            gallery_[name].push_back(e);
        std::cout << "[Gallery] \"" << name << "\"  "
                  << embs.size() << " embedding(s) loaded\n";
    }
    std::cout << "[Gallery] Total identities: " << gallery_.size() << "\n";
}

void FaceRecognitionSystem::registerFace(const std::string& label,
                                          const cv::Mat& img,
                                          const FaceDetection* det) {
    std::vector<float> emb;
    if (det)
        emb = recognizer_.getEmbeddingFromDet(img, *det);
    else
        emb = recognizer_.getEmbedding(img);
    gallery_[label].push_back(emb);
    std::cout << "[Gallery] Registered \"" << label << "\""
              << " (total samples: " << gallery_[label].size() << ")\n";
}

std::pair<std::string, float>
FaceRecognitionSystem::matchGallery(const std::vector<float>& emb) {
    std::string best_label;
    float       best_score = -1.f;
    for (auto& [label, embs] : gallery_) {
        for (auto& e : embs) {
            float s = EdgeFace::cosineSimilarity(emb, e);
            if (s > best_score) { best_score = s; best_label = label; }
        }
    }
    if (best_score >= rec_threshold_)
        return {best_label, best_score};
    return {"", best_score};
}

std::vector<FaceResult>
FaceRecognitionSystem::processFrame(const cv::Mat& img, float det_thresh,
                                     cv::Size input_size) {
    frame_idx_++;

    // 1. Detect
    std::vector<FaceDetection> face_dets;
    detector_.detect(img, face_dets, det_thresh, input_size);

    // 2. Convert detections to tracker format [x1,y1,x2,y2,score]
    std::vector<std::array<float, 5>> bboxes;
    for (auto& d : face_dets)
        bboxes.push_back({d.x1, d.y1, d.x2, d.y2, d.score});

    // 3. Track (always update even when empty)
    auto track_results = tracker_.update(bboxes);
    if (track_results.empty()) return {};

    // 4. Map each confirmed track → nearest raw detection by IoU
    std::map<int, int> tid_to_det;
    for (auto& tr : track_results) {
        int   best_di  = -1;
        float best_iou = 0.f;
        for (int di = 0; di < (int)bboxes.size(); di++) {
            float v = IoUTracker::iou(tr.bbox, bboxes[di]);
            if (v > best_iou) { best_iou = v; best_di = di; }
        }
        tid_to_det[tr.track_id] = (best_iou > 0.3f) ? best_di : -1;
    }

    // 5. Recognise (selective: new track, no cache, or interval elapsed)
    std::vector<FaceResult> outputs;
    for (auto& tr : track_results) {
        int  det_idx = tid_to_det[tr.track_id];
        auto* cached = track_cache_.count(tr.track_id)
                       ? &track_cache_[tr.track_id] : nullptr;

        bool need_rec = tr.is_new
                     || !cached
                     || (frame_idx_ - cached->frame) >= re_id_interval_;

        std::vector<float> emb;
        std::string        label;
        float              score = 0.f;

        if (need_rec && det_idx >= 0) {
            emb              = recognizer_.getEmbeddingFromDet(img, face_dets[det_idx]);
            auto [lbl, sc]   = matchGallery(emb);
            label            = lbl;
            score            = sc;
            track_cache_[tr.track_id] = {emb, label, score, frame_idx_};
        } else if (cached) {
            emb   = cached->embedding;
            label = cached->label;
            score = cached->rec_score;
        }

        // Update tracker label/embedding
        if (tracker_.tracks.count(tr.track_id)) {
            tracker_.tracks.at(tr.track_id).label     = label;
            tracker_.tracks.at(tr.track_id).embedding = emb;
        }

        outputs.push_back({tr.track_id, tr.bbox, emb, label, score, tr.is_new});
    }
    return outputs;
}

cv::Mat FaceRecognitionSystem::draw(const cv::Mat& img,
                                     const std::vector<FaceResult>& results,
                                     float fps, float avg_fps) {
    cv::Mat vis = img.clone();

    for (auto& r : results) {
        int x1 = (int)r.bbox[0], y1 = (int)r.bbox[1];
        int x2 = (int)r.bbox[2], y2 = (int)r.bbox[3];

        cv::Scalar color = r.label.empty()
                         ? cv::Scalar(0, 0, 255)    // Unknown - red
                         : cv::Scalar(0, 255, 0);   // Known   - green

        cv::rectangle(vis, {x1, y1}, {x2, y2}, color, 2);

        char buf[64];
        std::snprintf(buf, sizeof(buf), "ID:%d %s (%.2f)",
                      r.track_id,
                      r.label.empty() ? "Unknown" : r.label.c_str(),
                      r.rec_score);
        cv::putText(vis, buf, {x1, std::max(y1 - 8, 12)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2);
    }

    // FPS overlay
    char fps_buf[64], avg_buf[64];
    std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %6.1f", fps);
    std::snprintf(avg_buf, sizeof(avg_buf), "Avg FPS: %6.1f", avg_fps);
    cv::putText(vis, fps_buf, {10, 30},  cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0,255,255), 2);
    cv::putText(vis, avg_buf, {10, 62},  cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0,200,255), 2);

    return vis;
}

#include <gst/gst.h>
#include <gst/app/app.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gst/gst.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <numeric>

#include "rknpu2/scrfd_rknn.h"
#include "common.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"
#include "edgeface.hpp"

static GstElement *appsrc = NULL;
static gboolean caps_set = FALSE;
rknn_app_context_t rknn_app_ctx;
float conf_thresh = 0.4f;
static float rec_threshold_ = 0.5f;

static uint64_t frame_count = 0;
static double fps_sum = 0.0;
static uint64_t fps_count = 0;
static struct timespec last_time;
static gboolean started_fps = FALSE;

static EdgeFace* edgeface = nullptr;
static std::map<std::string, std::vector<std::vector<float>>> gallery_;

static const char FACEDB_MAGIC[8] = {'F','A','C','E','D','B','\0','\0'};

std::map<std::string, std::vector<std::vector<float>>> loadFaceDb(const std::string& db_path) {
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

std::pair<std::string, float> matchGallery(const std::vector<float>& emb) {
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
    return {"Unknown", best_score};
}

static GstFlowReturn on_new_sample(GstElement *sink, gpointer)
{
    GstSample *sample = gst_app_sink_pull_sample(GST_APP_SINK(sink));
    if (!sample) return GST_FLOW_OK;

    if (!caps_set) {
        gst_app_src_set_caps(
            GST_APP_SRC(appsrc),
            gst_sample_get_caps(sample)
        );
        caps_set = TRUE;
    }

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *s = gst_caps_get_structure(caps, 0);

    int width = 0, height = 0;
    gst_structure_get_int(s, "width", &width);
    gst_structure_get_int(s, "height", &height);
    const gchar *fmt = gst_structure_get_string(s, "format");

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(src_image));
    src_image.width = width;
    src_image.height = height;
    src_image.width_stride = width;
    src_image.height_stride = height;
    src_image.size = map.size;
    src_image.fd = -1;

    unsigned char *local_buf = (unsigned char *)malloc(map.size);
    memcpy(local_buf, map.data, map.size);
    src_image.virt_addr = local_buf;

    if (strcmp(fmt, "RGB") == 0) {
        src_image.format = IMAGE_FORMAT_RGB888;
    } else if (strcmp(fmt, "RGBA") == 0) {
        src_image.format = IMAGE_FORMAT_RGBA8888;
    } else {
        // handle other formats as needed
        gst_buffer_unmap(buffer, &map);
        free(local_buf);
        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    // --- Resize for detection ---
    image_buffer_t det_image;
    memset(&det_image, 0, sizeof(det_image));
    det_image.width = 640;
    det_image.height = 640;
    det_image.format = src_image.format;
    det_image.fd = -1;
    det_image.size = get_image_size(&det_image);
    det_image.virt_addr = (uint8_t*)malloc(det_image.size);

    convert_image(&src_image, &det_image, NULL, NULL, 0);

    object_detect_result_list od_results;
    int ret = inference_scrfd_model(&rknn_app_ctx, &det_image, &od_results, conf_thresh);

    // free resized buffer
    free(det_image.virt_addr);

    if (ret == 0) {
        for (int i = 0; i < od_results.count; i++) {
            object_detect_result *det = &(od_results.results[i]);

            float scale_x = (float)src_image.width / det_image.width;
            float scale_y = (float)src_image.height / det_image.height;

            int x1 = (int)(det->box.left * scale_x);
            int y1 = (int)(det->box.top * scale_y);
            int x2 = (int)(det->box.right * scale_x);
            int y2 = (int)(det->box.bottom * scale_y);

            draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

            // Convert to cv::Mat
            cv::Mat frame(src_image.height, src_image.width,
                (src_image.format == IMAGE_FORMAT_RGB888) ? CV_8UC3 : CV_8UC4,
                src_image.virt_addr);

            FaceDetection fd;
            fd.x1 = x1;
            fd.y1 = y1;
            fd.x2 = x2;
            fd.y2 = y2;
            for (int j = 0; j < 5; j++) {
                fd.kps[j][0] = det->kps[j].x * scale_x;
                fd.kps[j][1] = det->kps[j].y * scale_y;
            }
            // Extract embedding
            std::vector<float> emb = edgeface->getEmbeddingFromDet(frame, fd);

            auto [lbl, sc] = matchGallery(emb);
            char text[64];
            sprintf(text, "%s %.2f", lbl.c_str(), sc * 100);
            if (strcmp(lbl.c_str(), "Unknown") == 0) {
                draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
            } else {
                draw_text(&src_image, text, x1, y1 - 20, COLOR_GREEN, 10);
            }

            // for (int j = 0; j < 5; j++) {
            //     draw_circle(&src_image, int(det->kps[j].x * scale_x), int(det->kps[j].y * scale_y), 5, COLOR_GREEN, 2);
            // }
        }
    }

    // --- FPS calculation and overlay ---
    frame_count++;
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);

    if (frame_count == 100) {
        last_time = now;
        started_fps = TRUE;
    }

    if (started_fps) {
        double elapsed = (now.tv_sec - last_time.tv_sec) +
                         (now.tv_nsec - last_time.tv_nsec) / 1e9;
        if (elapsed > 0) {
            double fps = 1.0 / elapsed;
            fps_sum += fps;
            fps_count++;
            double avg_fps = fps_sum / fps_count;

            char fps_text[64];
            sprintf(fps_text, "FPS: %.2f (avg %.2f)", fps, avg_fps);
            draw_text(&src_image, fps_text, 20, 40, COLOR_GREEN, 14);
        }
        last_time = now;
    }

    // --- Push original-sized frame ---
    GstBuffer *copy = gst_buffer_new_allocate(NULL, src_image.size, NULL);
    GstMapInfo out_map;
    gst_buffer_map(copy, &out_map, GST_MAP_WRITE);
    memcpy(out_map.data, src_image.virt_addr, src_image.size);
    gst_buffer_unmap(copy, &out_map);

    GST_BUFFER_PTS(copy) = GST_BUFFER_PTS(buffer);
    GST_BUFFER_DTS(copy) = GST_BUFFER_DTS(buffer);
    GST_BUFFER_DURATION(copy) = GST_BUFFER_DURATION(buffer);

    GstFlowReturn push_ret;
    g_signal_emit_by_name(appsrc, "push-buffer", copy, &push_ret);

    gst_buffer_unref(copy);
    gst_buffer_unmap(buffer, &map);
    gst_sample_unref(sample);
    free(local_buf);

    return GST_FLOW_OK;
}

int main(int argc, char *argv[])
{
    gst_init(&argc, &argv);

    const char *input = (argc <= 1) ? "/home/hw/data/videos/test_1280_720_60fps.mp4" : argv[1];
    const char *face_db = (argc <= 2) ? "/home/hw/data/embeddings/face_db_7_5.bin" : argv[2];
    const char *host = (argc <= 3) ? "10.0.0.1" : argv[3];
    const char *model_path = "/home/hw/models/scrfd/scrfd_500m_bnkps_shape640x640.rknn";

    int ret;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_scrfd_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_scrfd_model fail! ret=%d model_path=%s\n", ret, model_path);
        ret = release_scrfd_model(&rknn_app_ctx);
    }
    edgeface = new EdgeFace("/home/hw/models/edgeface/edgeface_xs_gamma_06.onnx");

    auto loaded = loadFaceDb(face_db);
    float sum = 0.0f;
    for (auto& [name, embs] : loaded) {
        for (auto& e : embs) {
            gallery_[name].push_back(e);
            // sum = std::accumulate(e.begin(), e.end(), 0.0f);
            // printf("Sum: %.2f", sum);
        }
        std::cout << "[Gallery] \"" << name << "\"  "
                  << embs.size() << " embedding(s) loaded \n";
    }
    std::cout << "[Gallery] Total identities: " << gallery_.size() << "\n";

    gchar *src_desc = g_strdup_printf(
        "filesrc location=%s ! "
        "qtdemux name=demux demux.video_0 ! queue ! "
        "h264parse ! avdec_h264 ! videoconvert ! "
        "video/x-raw,format=RGB ! "
        "appsink name=mysink emit-signals=true sync=true",
        input
    );

    GstElement *src_pipeline = gst_parse_launch(src_desc, NULL);
    g_free(src_desc);

    g_signal_connect(
        gst_bin_get_by_name(GST_BIN(src_pipeline), "mysink"),
        "new-sample",
        G_CALLBACK(on_new_sample),
        NULL
    );

    gchar *send_desc = g_strdup_printf(
        "appsrc name=mysrc is-live=true format=time do-timestamp=false ! "
        "queue ! videoconvert ! "
        // "x264enc tune=zerolatency bitrate=2048 speed-preset=ultrafast ! "
        "x264enc tune=zerolatency bitrate=8000 speed-preset=veryfast ! "
        "h264parse config-interval=-1 ! "
        "rtph264pay pt=96 config-interval=1 ! "
        "udpsink host=%s port=5000 sync=true async=false",
        host
    );

    GstElement *send_pipeline = gst_parse_launch(send_desc, NULL);
    g_free(send_desc);

    appsrc = gst_bin_get_by_name(GST_BIN(send_pipeline), "mysrc");

    g_object_set(appsrc,
                 "is-live", TRUE,
                 "format", GST_FORMAT_TIME,
                 "block", TRUE,
                 NULL);

    GstClock *clock = gst_system_clock_obtain();
    gst_element_set_clock(src_pipeline, clock);
    gst_element_set_clock(send_pipeline, clock);
    gst_object_unref(clock);

    gst_element_set_state(src_pipeline, GST_STATE_PLAYING);
    gst_element_set_state(send_pipeline, GST_STATE_PLAYING);

    g_print("[Start] Streaming RTP to %s:5000...\n", host);

    GstBus *bus = gst_element_get_bus(src_pipeline);
    GstMessage *msg = gst_bus_timed_pop_filtered(
        bus,
        GST_CLOCK_TIME_NONE,
        static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS)
    );

    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS)
        g_print("[Info] Average FPS: %.2f\n", fps_sum / fps_count);
        g_print("[Finish] End of stream!\n");

    if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR)
        g_printerr("Re-run with the GST_DEBUG=*:WARN environment variable set for more details.\n");

    gst_message_unref(msg);
    gst_object_unref(bus);

    gst_element_set_state(src_pipeline, GST_STATE_NULL);
    gst_element_set_state(send_pipeline, GST_STATE_NULL);

    gst_object_unref(src_pipeline);
    gst_object_unref(send_pipeline);

    return 0;
}
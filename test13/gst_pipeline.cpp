#include "gst_pipeline.hpp"
#include <iostream>
#include <stdexcept>

FaceGstPipeline::FaceGstPipeline(const std::string& uri, FrameCallback cb,
                                 int width, int height)
    : cb_(std::move(cb))
{
    std::string caps = "video/x-raw,format=BGR";
    if (width > 0 && height > 0)
        caps += ",width=" + std::to_string(width)
              + ",height=" + std::to_string(height);

    std::string src;
    if (uri.rfind("rtsp://", 0) == 0)
        src = "rtspsrc location=" + uri + " latency=0 ! rtph264depay ! h264parse ! avdec_h264";
    else if (uri.rfind("/dev/video", 0) == 0)
        src = "v4l2src device=" + uri;
    else
        src = "filesrc location=" + uri + " ! decodebin";

    std::string desc =
        src +
        " ! videoconvert"
        " ! " + caps +
        " ! appsink name=sink max-buffers=1 drop=true sync=false";

    std::cout << "[GST] Pipeline: " << desc << "\n";

    GError* err = nullptr;
    pipeline_   = gst_parse_launch(desc.c_str(), &err);
    if (!pipeline_) {
        std::string msg = err ? err->message : "unknown";
        g_error_free(err);
        throw std::runtime_error("[GST] Failed to create pipeline: " + msg);
    }

    GstAppSink* appsink =
        GST_APP_SINK(gst_bin_get_by_name(GST_BIN(pipeline_), "sink"));
    if (!appsink)
        throw std::runtime_error("[GST] Cannot find appsink element");

    GstAppSinkCallbacks cbs{};
    cbs.new_sample = onNewSample;
    gst_app_sink_set_callbacks(appsink, &cbs, this, nullptr);
    gst_app_sink_set_emit_signals(appsink, false);
    gst_object_unref(appsink);
}

FaceGstPipeline::~FaceGstPipeline() {
    stop();
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
    }
}

void FaceGstPipeline::run() {
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    std::cout << "[GST] Pipeline playing. Press 'q' to quit.\n";

    GstBus* bus = gst_element_get_bus(pipeline_);
    while (!should_exit_) {
        GstMessage* msg = gst_bus_timed_pop_filtered(
            bus, 100 * GST_MSECOND,
            static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
        if (!msg) continue;

        switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            std::cout << "[GST] End of stream.\n";
            should_exit_ = true;
            break;
        case GST_MESSAGE_ERROR: {
            GError* err = nullptr;
            gst_message_parse_error(msg, &err, nullptr);
            std::cerr << "[GST] Error: " << (err ? err->message : "unknown") << "\n";
            g_error_free(err);
            should_exit_ = true;
            break;
        }
        default: break;
        }
        gst_message_unref(msg);
    }
    gst_object_unref(bus);
}

void FaceGstPipeline::stop() { should_exit_ = true; }

GstFlowReturn FaceGstPipeline::onNewSample(GstAppSink* sink, gpointer user_data) {
    auto* self = static_cast<FaceGstPipeline*>(user_data);
    if (self->should_exit_) return GST_FLOW_EOS;

    GstSample* sample = gst_app_sink_pull_sample(sink);
    if (!sample) return GST_FLOW_OK;

    GstBuffer*    buf  = gst_sample_get_buffer(sample);
    GstCaps*      caps = gst_sample_get_caps(sample);
    GstStructure* st   = gst_caps_get_structure(caps, 0);

    int w = 0, h = 0;
    gst_structure_get_int(st, "width",  &w);
    gst_structure_get_int(st, "height", &h);

    GstMapInfo map;
    if (gst_buffer_map(buf, &map, GST_MAP_READ)) {
        cv::Mat frame(h, w, CV_8UC3, map.data);
        cv::Mat frame_copy = frame.clone();
        gst_buffer_unmap(buf, &map);
        self->cb_(frame_copy);
    }

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

#include "gst_streamer.hpp"
#include <stdexcept>
#include <sstream>

static std::string build_pipeline_description(const std::string& host,
                                              int port,
                                              int width,
                                              int height,
                                              int fps,
                                              int bitrate) {
    std::ostringstream ss;
    ss << "appsrc name=src is-live=true block=true format=time caps=video/x-raw,format=BGR,width="
       << width << ",height=" << height << ",framerate=" << fps << "/1"
       << " ! videoconvert"
       << " ! x264enc tune=zerolatency speed-preset=ultrafast bitrate=" << bitrate
       << " key-int-max=" << (fps * 2)
       << " ! video/x-h264,profile=baseline"
       << " ! h264parse"
       << " ! rtph264pay config-interval=1 pt=96"
       << " ! udpsink host=" << host << " port=" << port
       << " sync=false async=false";
    return ss.str();
}

GstRtpStreamer::GstRtpStreamer(const std::string& host, int port,
                               int width, int height, int fps, int bitrate)
    : fps_(fps)
{
    if (host.empty())
        throw std::runtime_error("Invalid RTP host");
    if (port <= 0)
        throw std::runtime_error("Invalid RTP port");
    if (width <= 0 || height <= 0)
        throw std::runtime_error("Invalid video dimensions");
    if (fps <= 0)
        throw std::runtime_error("Invalid frame rate");

    gst_init(nullptr, nullptr);
    std::string pipeline_desc = build_pipeline_description(host, port, width, height, fps, bitrate);

    GError* err = nullptr;
    pipeline_ = gst_parse_launch(pipeline_desc.c_str(), &err);
    if (!pipeline_) {
        std::string msg = err ? err->message : "unknown pipeline error";
        g_error_free(err);
        throw std::runtime_error("[GST] Failed to create pipeline: " + msg);
    }

    appsrc_ = gst_bin_get_by_name(GST_BIN(pipeline_), "src");
    if (!appsrc_) {
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        throw std::runtime_error("[GST] Cannot find appsrc element");
    }

    g_object_set(appsrc_, "format", GST_FORMAT_TIME, nullptr);
    gst_element_set_state(pipeline_, GST_STATE_PLAYING);
}

GstRtpStreamer::~GstRtpStreamer() {
    stop();
    if (appsrc_) {
        gst_object_unref(appsrc_);
        appsrc_ = nullptr;
    }
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
    }
}

void GstRtpStreamer::pushFrame(const cv::Mat& frame) {
    if (!pipeline_ || !appsrc_)
        throw std::runtime_error("[GST] RTP streamer is not initialized");
    if (frame.empty())
        return;
    if (frame.type() != CV_8UC3)
        throw std::runtime_error("[GST] Frame must be CV_8UC3 BGR format");

    const size_t size = frame.total() * frame.elemSize();
    GstBuffer* buffer = gst_buffer_new_allocate(nullptr, size, nullptr);
    if (!buffer)
        throw std::runtime_error("[GST] Failed to allocate buffer");

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_WRITE)) {
        gst_buffer_unref(buffer);
        throw std::runtime_error("[GST] Failed to map buffer");
    }
    std::memcpy(map.data, frame.data, size);
    gst_buffer_unmap(buffer, &map);

    GST_BUFFER_PTS(buffer) = timestamp_;
    GST_BUFFER_DURATION(buffer) = gst_util_uint64_scale_int(1, GST_SECOND, fps_);
    timestamp_ += GST_BUFFER_DURATION(buffer);

    GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(appsrc_), buffer);
    if (ret != GST_FLOW_OK) {
        throw std::runtime_error("[GST] Failed to push buffer to appsrc");
    }
}

void GstRtpStreamer::stop() {
    if (appsrc_) {
        gst_app_src_end_of_stream(GST_APP_SRC(appsrc_));
    }
}

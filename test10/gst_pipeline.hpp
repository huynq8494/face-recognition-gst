#pragma once
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <opencv2/opencv.hpp>
#include <functional>
#include <string>
#include <atomic>

using FrameCallback = std::function<void(const cv::Mat& frame)>;

class FaceGstPipeline {
public:
    FaceGstPipeline(const std::string& uri, FrameCallback cb,
                    int width = 0, int height = 0);
    ~FaceGstPipeline();

    void run();
    void stop();

private:
    GstElement*       pipeline_ = nullptr;
    FrameCallback     cb_;
    std::atomic<bool> should_exit_{false};

    static GstFlowReturn onNewSample(GstAppSink* sink, gpointer user_data);
};

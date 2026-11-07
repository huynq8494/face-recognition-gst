#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <opencv2/opencv.hpp>
#include <string>

class GstRtpStreamer {
public:
    GstRtpStreamer(const std::string& host, int port,
                   int width, int height, int fps = 25, int bitrate = 1200);
    ~GstRtpStreamer();

    void pushFrame(const cv::Mat& frame);
    void stop();

private:
    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GstClockTime timestamp_ = 0;
    int fps_ = 25;
};

// GStreamer Video + YOLO Object Detection Integration
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/video/video.h>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <array>
#include <cstring>
#include <atomic>
#include <thread>

// Global YOLO model path
const char* MODEL_PATH = "/home/huynq/models/yolo/yolo26n.onnx";

// Global variables for FPS calculation
static std::atomic<int> g_frame_count(0);
static std::atomic<int> g_processed_count(0);
static double g_total_fps = 0.0;

// YOLO Detection result
struct DetectionResult {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};

// Global state
static std::atomic<bool> should_exit(false);
static Ort::Session* g_session = nullptr;
static Ort::Env* g_env = nullptr;

// Preprocess: resize, transpose (HWC->CHW), normalize
std::vector<float> preprocess(const cv::Mat& img) {
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(640, 640));
    resized.convertTo(resized, CV_32F, 1.0 / 255.0);

    std::vector<float> input_tensor_values;
    input_tensor_values.reserve(1 * 3 * 640 * 640);

    // HWC -> CHW
    for (int c = 0; c < 3; c++) {
        for (int y = 0; y < 640; y++) {
            for (int x = 0; x < 640; x++) {
                input_tensor_values.push_back(resized.at<cv::Vec3f>(y, x)[c]);
            }
        }
    }
    return input_tensor_values;
}

// Run YOLO detection on input image
std::vector<DetectionResult> detect_objects(const cv::Mat& img) {
    std::vector<DetectionResult> results;
    
    if (!g_session) return results;

    auto input_tensor_values = preprocess(img);
    std::array<int64_t, 4> input_shape{ 1, 3, 640, 640 };

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_tensor_values.data(), input_tensor_values.size(),
        input_shape.data(), input_shape.size());

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = g_session->GetInputNameAllocated(0, allocator);
    auto output_name_alloc = g_session->GetOutputNameAllocated(0, allocator);

    const char* input_name = input_name_alloc.get();
    const char* output_name = output_name_alloc.get();

    auto output_tensors = g_session->Run(
        Ort::RunOptions{ nullptr },
        &input_name, &input_tensor, 1,
        &output_name, 1);

    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    auto output_shape = output_tensors.front().GetTensorTypeAndShapeInfo().GetShape();

    int num_boxes = static_cast<int>(output_shape[1]);
    int attributes = static_cast<int>(output_shape[2]);

    // Get original image dimensions
    int h = img.rows, w = img.cols;

    // Loop over detections
    for (int i = 0; i < num_boxes; i++) {
        float x1 = output_data[i * attributes + 0];
        float y1 = output_data[i * attributes + 1];
        float x2 = output_data[i * attributes + 2];
        float y2 = output_data[i * attributes + 3];
        float conf = output_data[i * attributes + 4];
        int cls_id = static_cast<int>(output_data[i * attributes + 5]);

        if (conf > 0.25f) {
            DetectionResult det;
            // Scale back to original image size
            det.x1 = x1 * w / 640.0f;
            det.y1 = y1 * h / 640.0f;
            det.x2 = x2 * w / 640.0f;
            det.y2 = y2 * h / 640.0f;
            det.confidence = conf;
            det.class_id = cls_id;
            results.push_back(det);
        }
    }

    return results;
}

// GStreamer appsink callback
static GstFlowReturn on_new_sample(GstAppSink *appsink, gpointer user_data) {
    g_frame_count++;
    auto start = std::chrono::high_resolution_clock::now();

    GstSample *sample = gst_app_sink_pull_sample(appsink);
    if (!sample) return GST_FLOW_OK;

    GstBuffer *buffer = gst_sample_get_buffer(sample);
    GstCaps *caps = gst_sample_get_caps(sample);
    GstStructure *structure = gst_caps_get_structure(caps, 0);

    int width, height;
    gst_structure_get_int(structure, "width", &width);
    gst_structure_get_int(structure, "height", &height);

    GstMapInfo map;
    gst_buffer_map(buffer, &map, GST_MAP_READ);

    // Create Mat from GStreamer buffer (assuming RGB24 format)
    cv::Mat frame(height, width, CV_8UC3, map.data);
    cv::Mat frame_copy = frame.clone();

    gst_buffer_unmap(buffer, &map);

    // Run detection
    auto detections = detect_objects(frame_copy);

    // Draw detections
    for (const auto& det : detections) {
        int x1 = static_cast<int>(det.x1);
        int y1 = static_cast<int>(det.y1);
        int x2 = static_cast<int>(det.x2);
        int y2 = static_cast<int>(det.y2);

        cv::rectangle(frame_copy, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
        cv::putText(frame_copy, "ID:" + std::to_string(det.class_id) + " " + 
                    std::to_string(det.confidence).substr(0, 4),
                    cv::Point(x1, y1 - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 1);
    }

    // FPS calculation only after 100 frames
    if (g_frame_count > 100) {
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(end - start).count();
        double fps = 1.0 / elapsed;

        g_processed_count++;
        g_total_fps += fps;
        double avg_fps = g_total_fps / g_processed_count;

        // Overlay FPS text on frame
        cv::putText(frame_copy,
                    "FPS: " + std::to_string(fps).substr(0,5) +
                    " | Avg: " + std::to_string(avg_fps).substr(0,5),
                    cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 255, 255), 2);
    }

    // Display
    cv::imshow("YOLO Object Detection - GStreamer", frame_copy);
    int key = cv::waitKey(1);
    if (key == 'q' || key == 27) {  // 'q' or ESC to quit
        should_exit = true;
    }

    gst_sample_unref(sample);
    return GST_FLOW_OK;
}

int main(int argc, char *argv[]) {
    GstElement *pipeline;
    GstBus *bus;
    GstMessage *msg;

    // Initialize GStreamer
    gst_init(&argc, &argv);

    // Initialize ONNX Runtime
    g_env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "yolo");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    try {
        g_session = new Ort::Session(*g_env, MODEL_PATH, session_options);
        std::cout << "YOLO model loaded successfully from: " << MODEL_PATH << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load YOLO model: " << e.what() << std::endl;
        return 1;
    }

    // Build the pipeline with appsink for frame extraction
    const char *launch_string = "filesrc location=/home/huynq/data/videos/test_1280_720_60fps.mp4 ! "
                                "decodebin ! "
                                "videoflip method=counterclockwise ! "
                                "videoconvert ! "
                                "video/x-raw,format=BGR ! "
                                "appsink name=sink";

    GError *error = nullptr;
    pipeline = gst_parse_launch(launch_string, &error);

    if (!pipeline) {
        std::cerr << "Failed to create pipeline: " << error->message << std::endl;
        g_error_free(error);
        return 1;
    }

    // Configure appsink
    GstAppSink *appsink = GST_APP_SINK(gst_bin_get_by_name(GST_BIN(pipeline), "sink"));
    GstAppSinkCallbacks callbacks;
    memset(&callbacks, 0, sizeof(callbacks));
    callbacks.new_sample = on_new_sample;
    gst_app_sink_set_callbacks(appsink, &callbacks, nullptr, nullptr);
    gst_app_sink_set_emit_signals(appsink, false);
    gst_object_unref(appsink);

    // Start playing
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    std::cout << "Pipeline started. Press 'q' or ESC to exit." << std::endl;

    // Wait until error or EOS or user wants to exit
    bus = gst_element_get_bus(pipeline);
    while (!should_exit) {
        msg = gst_bus_timed_pop_filtered(bus, 100 * GST_MSECOND,
                                         static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

        if (msg) {
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                g_printerr("An error occurred! Re-run with the GST_DEBUG=*:WARN environment variable set for more details.\n");
            }
            gst_message_unref(msg);
            break;
        }
    }

    // Cleanup
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(bus);
    gst_object_unref(pipeline);

    delete g_session;
    delete g_env;

    cv::destroyAllWindows();
    std::cout << "Cleanup completed." << std::endl;

    if (g_processed_count > 0) {
      std::cout << "Final Average FPS: " << g_total_fps / g_processed_count << std::endl;
    }

    return 0;
}

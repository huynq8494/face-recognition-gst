#include <gst/gst.h>
#include <gst/app/app.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gst/gst.h>

#include "rknpu2/scrfd_rknn.h"
#include "common.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

static GstElement *appsrc = NULL;
static gboolean caps_set = FALSE;
rknn_app_context_t rknn_app_ctx;
float conf_thresh = 0.4f;

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
            char text[64];
            sprintf(text, "face %.1f%%", det->prop * 100);
            draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
        }
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
    const char *host = (argc <= 2) ? "10.0.0.1" : argv[2];
    const char *model_path = "/home/hw/models/scrfd/scrfd_500m_bnkps_shape640x640.rknn";
    // const char *image_path = (argc <= 1) ? "/home/hw/data/images/scrfd_test.jpg" : argv[1];
    // float conf_thresh = (argc <= 2) ? 0.4f : std::atof(argv[2]);
    
    int ret;    
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_scrfd_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_scrfd_model fail! ret=%d model_path=%s\n", ret, model_path);
        ret = release_scrfd_model(&rknn_app_ctx);
    }
    
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
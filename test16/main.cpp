#include <gst/gst.h>
#include <gst/app/app.h>

static GstElement *appsrc = NULL;
static gboolean caps_set = FALSE;

/* appsink → appsrc */
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
    GstBuffer *copy = gst_buffer_copy(buffer);

    GST_BUFFER_PTS(copy) = GST_BUFFER_PTS(buffer);
    GST_BUFFER_DTS(copy) = GST_BUFFER_DTS(buffer);
    GST_BUFFER_DURATION(copy) = GST_BUFFER_DURATION(buffer);

    GstFlowReturn ret;
    g_signal_emit_by_name(appsrc, "push-buffer", copy, &ret);

    gst_buffer_unref(copy);
    gst_sample_unref(sample);

    return GST_FLOW_OK;
}

int main(int argc, char *argv[])
{
    gst_init(&argc, &argv);

    const char *input = (argc <= 1) ? "/home/hw/data/videos/test_1280_720_60fps.mp4" : argv[1];
    const char *host = (argc <= 2) ? "10.0.0.1" : argv[2];

    gchar *src_desc = g_strdup_printf(
        "filesrc location=%s ! "
        "qtdemux name=demux demux.video_0 ! queue ! "
        "h264parse config-interval=-1 ! "
        "video/x-h264,stream-format=byte-stream,alignment=au ! "
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
        "queue ! h264parse config-interval=-1 ! "
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
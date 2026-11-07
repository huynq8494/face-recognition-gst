#include <gst/gst.h>
#include <cairo/cairo.h>
#include <cstdlib>
#include <ctime>
#include <glib.h>

static void draw_overlay(GstElement *overlay, cairo_t *cr,
                         guint64 timestamp, guint64 duration, gpointer user_data)
{
    int width = 1280, height = 720;
    int rw = 100 + rand() % 200;
    int rh = 100 + rand() % 200;
    int rx = rand() % (width - rw);
    int ry = rand() % (height - rh);

    cairo_set_source_rgba(cr,
        (double)rand()/RAND_MAX,
        (double)rand()/RAND_MAX,
        (double)rand()/RAND_MAX,
        0.5);
    cairo_rectangle(cr, rx, ry, rw, rh);
    cairo_fill(cr);
}

int main(int argc, char *argv[])
{
    gst_init(&argc, &argv);
    srand(time(NULL));  // Seed random number generator for overlay rectangles

    GstElement *pipeline = gst_parse_launch(
        "filesrc location=/home/hw/data/videos/test_1280_720_60fps.mp4 ! "
        "qtdemux ! h264parse ! avdec_h264 ! "
        "videoconvert ! cairooverlay name=overlay ! "
        "videoconvert ! x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! "
        "rtph264pay pt=96 config-interval=1 ! "
        "udpsink host=10.0.0.1 port=5000",
        NULL);

    GstElement *overlay = gst_bin_get_by_name(GST_BIN(pipeline), "overlay");
    g_signal_connect(overlay, "draw", G_CALLBACK(draw_overlay), NULL);

      /* Start playing */
    gst_element_set_state (pipeline, GST_STATE_PLAYING);

    /* Wait until error or EOS */
    GstBus *bus = gst_element_get_bus (pipeline);
    GstMessage *msg = gst_bus_timed_pop_filtered (bus, GST_CLOCK_TIME_NONE, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
    if (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_EOS) {
        g_print ("End-Of-Stream reached.\n");
    }
    if (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_ERROR) {
        g_printerr ("An error occurred! Re-run with the GST_DEBUG=*:WARN environment variable set for more details.\n");
    }

    // Cleanup
    gst_message_unref (msg);
    gst_object_unref (bus);
    gst_element_set_state (pipeline, GST_STATE_NULL);
    gst_object_unref (pipeline);
    return 0;
}

#include <gst/gst.h>

static void pad_added_handler(GstElement *src, GstPad *new_pad, gpointer data) {
    GstElement *parser = static_cast<GstElement*>(data);
    GstPad *sink_pad = gst_element_get_static_pad(parser, "sink");

    if (gst_pad_is_linked(sink_pad)) {
        gst_object_unref(sink_pad);
        return;
    }

    if (gst_pad_link(new_pad, sink_pad) != GST_PAD_LINK_OK) {
        g_printerr("Failed to link demux to parser.\n");
    }
    gst_object_unref(sink_pad);
}

int main(int argc, char *argv[]) {
    gst_init(&argc, &argv);

    // Create pipeline and elements
    GstElement *pipeline = gst_pipeline_new("sender-pipeline");
    GstElement *source   = gst_element_factory_make("filesrc", "source");
    GstElement *demux    = gst_element_factory_make("qtdemux", "demux");
    GstElement *parser   = gst_element_factory_make("h264parse", "parser");
    GstElement *pay      = gst_element_factory_make("rtph264pay", "pay");
    GstElement *sink     = gst_element_factory_make("udpsink", "sink");

    if (!pipeline || !source || !demux || !parser || !pay || !sink) {
        g_printerr("Failed to create elements.\n");
        return -1;
    }

    // Configure properties
    g_object_set(source, "location", "/home/hw/data/videos/test_1280_720_60fps.mp4", NULL);
    g_object_set(pay, "pt", 96, "config-interval", 1, NULL); // send SPS/PPS regularly
    g_object_set(sink, "host", "10.0.0.1", "port", 5000, NULL);

    // Build pipeline
    gst_bin_add_many(GST_BIN(pipeline), source, demux, parser, pay, sink, NULL);

    if (!gst_element_link(source, demux)) {
        g_printerr("Source and demux could not be linked.\n");
        gst_object_unref(pipeline);
        return -1;
    }

    if (!gst_element_link_many(parser, pay, sink, NULL)) {
        g_printerr("Parser, pay, and sink could not be linked.\n");
        gst_object_unref(pipeline);
        return -1;
    }

    // Connect demuxer pad dynamically
    g_signal_connect(demux, "pad-added", G_CALLBACK(pad_added_handler), parser);

    // Start playing
    GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr("Unable to set pipeline to PLAYING.\n");
        gst_object_unref(pipeline);
        return -1;
    }

    // Run until error or EOS
    GstBus *bus = gst_element_get_bus(pipeline);
    GstMessage *msg = gst_bus_timed_pop_filtered(bus, GST_CLOCK_TIME_NONE,
                                                 (GstMessageType)(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));

    if (msg) gst_message_unref(msg);
    gst_object_unref(bus);

    // Cleanup
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);

    return 0;
}

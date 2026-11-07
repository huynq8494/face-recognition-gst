#include <gst/gst.h>

int main (int argc, char *argv[])
{
  GstElement *pipeline;
  GstBus *bus;
  GstMessage *msg;

  /* Initialize GStreamer */
  gst_init (&argc, &argv);

  const char *host = (argc <= 1) ? "10.0.0.1" : argv[1];

  /* Build the pipeline */
  gchar *pipeline_desc = g_strdup_printf(
    "filesrc location=/home/hw/data/videos/test_1280_720_60fps.mp4 ! "
    "qtdemux ! h264parse ! rtph264pay pt=96 config-interval=1 ! "
    "udpsink host=%s port=5000",
    host
  );
  pipeline = gst_parse_launch(pipeline_desc, NULL);
  g_free(pipeline_desc);

  /* Start playing */
  gst_element_set_state (pipeline, GST_STATE_PLAYING);

  /* Wait until error or EOS */
  bus = gst_element_get_bus (pipeline);
  msg = gst_bus_timed_pop_filtered (bus, GST_CLOCK_TIME_NONE, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
  
  if (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_EOS) {
    g_print ("End-Of-Stream reached.\n");
  }

  if (GST_MESSAGE_TYPE (msg) == GST_MESSAGE_ERROR) {
    g_printerr ("An error occurred! Re-run with the GST_DEBUG=*:WARN environment variable set for more details.\n");
  }

  /* Free resources */
  gst_message_unref (msg);
  gst_object_unref (bus);
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (pipeline);
  return 0;
}

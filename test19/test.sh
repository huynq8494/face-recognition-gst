gst-launch-1.0 filesrc location=/home/hw/data/videos/test_1280_720_60fps.mp4 ! \
    qtdemux ! h264parse ! mppvideodec ! video/x-raw,format=NV12,width=1280,height=720 ! \
    rgaconvert ! video/x-raw,format=NV12,width=2560,height=1440 ! mpph264enc ! \
    rtph264pay pt=96 config-interval=1 ! udpsink host=10.0.0.1 port=5000

| Test name | Target | Description | Notes |
|---|---|---|---|
| test01 | x86 | Hello wold GStreamer app build under Cmake | Cmake, GStreamer |
| test02 | x86 | Object Detection with OnnxRuntime | Cmake, Object detection, Onnxruntime |
| test03 | x86 | GStreamer with input mp4 video | GStreamer, mp4 |
| test04 | x86 | GStreamer with mp4 input video then Object dection |  |
| test05 | x86 | scrfd with image, python full flow |  |
| test06 | x86 | Full flow C++ |  |
| test07 | x86 | Full flow C++ GStreamer integrate |  |
| test08 | aarch64 | Full flow C++ GStreamer integrate |  |
| test09 | aarch64 | SCRFD with rknn |  |
| test10 | aarch64 | Full flow C++, GStreamer with scrfd rknn |  |
| test11 | aarch64 | SCRFD with rknn, video input |  |
| test12 | aarch64 | Full flow C++ with scrfd rknn |  |
| test13 | aarch64 | GStreamer integrate |  |
| test14 | aarch64 | GStreamer, video input, stream via RTP/UDP |  |
| test15 | aarch64 | test14 + cairooverlay (CPU) |  |
| test16 | aarch64 | test14 + appsink + appsrc |  |
| test17 | aarch64 | test14 + rgaconvert |  |
| test18 | aarch64 | test16 + test09 |  |
| test19 | aarch64 | test18 + face recognition |  |
| test20 | aarch64 | test19 + tracker |  |
| test1 | aarch64 |  |

# Notes
- Make sure the remote board IP (10.0.0.5) and the host IP (10.0.0.1) in remote.sdp have been updated
- Start below command in the host before run app on the remote board.
    ```bash
    ffplay -protocol_whitelist file,udp,rtp -i remote.sdp
    ```
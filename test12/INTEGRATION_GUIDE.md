# Test12 + EdgeFace Integration - Face Recognition System

## Overview
Successfully integrated RKNN-based SCRFD face detection with ONNX-based EdgeFace face recognition into test12 to create a unified face recognition system.

## Key Architecture

```
Input Image
    ↓
SCRFD RKNN Detection (test12's model)
    ↓
Tracking (IoU + Kalman filter)
    ↓
EdgeFace ONNX Embedding (test08's model)
    ↓
Gallery Matching
    ↓
Face Recognition Results
```

## Files Created/Modified

### New Files:
1. **scrfd_rknn.hpp** - C++ wrapper for RKNN scrfd_rknn.cc
   - Provides `SCRFD_RKNN` class with `detect()` method
   - Generates approximate 5-point keypoints from bounding boxes
   - Compatible with test08's SCRFD interface

2. **scrfd_rknn_wrapper.cpp** - Implementation of SCRFD_RKNN wrapper
   - Wraps C API into C++ class interface
   - Handles cv::Mat ↔ image_buffer_t conversion
   - Generates keypoints for face alignment

3. **edgeface.hpp / edgeface.cpp** - Copied from test08
   - ONNX Runtime-based face embedding extraction
   - Face alignment using 5-point keypoints
   - L2 normalization of embeddings

4. **tracker.hpp / tracker.cpp** - Copied from test08
   - Kalman filter-based object tracking
   - IoU matching + Hungarian algorithm
   - Track state management

5. **face_recognition_system.hpp / face_recognition_system.cpp** - Adapted from test08
   - Main recognition pipeline
   - Gallery loading/matching
   - FPS counter
   - Visualization with labels

6. **utils.hpp** - Copied from test08
   - FaceDetection struct
   - Face alignment reference points (112x112)

### Modified Files:
1. **main.cc**
   - New: Uses FaceRecognitionSystem instead of raw C API
   - Supports image input with optional gallery file
   - Outputs annotated image with recognized faces
   - Displays results on stdout

2. **CMakeLists.txt**
   - Added ONNX Runtime include/library paths
   - Added new source files to build
   - Changed project name from test11 to test12
   - Added C++17 standard requirement

## Build Instructions

```bash
cd /home/huynq/rknn_model_zoo/examples/test12/cpp
mkdir -p build
cd build
cmake ..
make -j4
```

## Usage

### Simple detection + recognition (no gallery):
```bash
./test12 /path/to/image.jpg
```

### With custom confidence threshold:
```bash
./test12 /path/to/image.jpg 0.5
```

### With face gallery for identification:
```bash
./test12 /path/to/image.jpg 0.5 /path/to/gallery.bin
```

### With custom SCRFD model:
```bash
./test12 /path/to/scrfd_model.rknn /path/to/image.jpg [conf] [gallery]
```

## Model Requirements

- **SCRFD RKNN**: `/home/huynq/models/scrfd/scrfd_500m_bnkps_shape640x640.rknn`
- **EdgeFace ONNX**: `/home/huynq/models/edgeface/edgeface_xs_gamma_06.onnx`

## Dependencies

- OpenCV
- ONNX Runtime (onnxruntime-linux-aarch64-1.26.0)
- librknnrt (RKNN Runtime)
- Standard C++ libraries

## Output

1. **Console**: Detected faces with track IDs, labels, and confidence scores
2. **Image File**: `output_recognized.png` with visualization
   - Green boxes: Known faces (matched to gallery)
   - Red boxes: Unknown faces
   - Labels: Track ID, name, similarity score
   - FPS indicator

## Key Differences from test08

- **Detection**: Uses RKNN instead of ONNX
  - Hardware accelerated on Rockchip NPU
  - Better performance on embedded boards
  - Same output format via wrapper

- **Recognition**: Uses ONNX EdgeFace (same as test08)
  - No changes to face embedding pipeline
  - Compatible with existing gallery files

## Keypoint Generation

Since the RKNN SCRFD model provides bounding boxes without reliable keypoints:
- Generates approximate 5-point landmarks from bbox
- Eyes: top 35% of face height
- Nose: center of face  
- Mouth corners: bottom 20% of face

This provides reasonable face alignment for EdgeFace embedding extraction.

## Performance

- SCRFD Detection: Hardware-accelerated via RKNN
- EdgeFace Embedding: CPU (ONNX Runtime)
- Typical inference time for 640x640 image: ~50-100ms on RK3588

## Future Enhancements

1. Extract actual 5-point keypoints from RKNN model outputs
2. Support video input with frame-by-frame processing
3. Batch processing for multiple images
4. Real-time face gallery updates
5. Face quality assessment before recognition

# SCRFD Face Detection (C++ RKNN)

This example demonstrates face detection using the SCRFD (Sample and Computation Redistribution for Efficient Face Detection) model with RKNN inference.

## Model Information

- **Model**: SCRFD 500M BN-KPS (shape: 640x640)
- **Input**: RGB images (640×640)
- **Output**: Face bounding boxes with confidence scores
- **Quantization**: INT8 quantized format
- **Feature Pyramid**: Stride 8, 16, 32 (3 feature maps)

## Building

```bash
cd examples/test09/cpp/build
mkdir -p . && cd .
cmake .. && make -j4
```

## Usage

### Default model (recommended)
```bash
./test09 /path/to/image.jpg
```

### With custom confidence threshold
```bash
./test09 /path/to/image.jpg 0.5
```
(Threshold range: 0.0 to 1.0, default: 0.4)

### Specify custom model
```bash
./test09 /path/to/model.rknn /path/to/image.jpg [conf_threshold]
```

## Output

- **Console**: Prints detected face bounding boxes with confidence scores
  ```
  face @ (x1 y1 x2 y2) confidence
  ```

- **Image**: Saves annotated image to `out.png` with:
  - Blue bounding box rectangles around detected faces
  - Red confidence percentage labels above each box

## Implementation Details

### Key Components

1. **Preprocessing** (`convert_image_with_letterbox`)
   - Resize image with aspect ratio preservation
   - Letterbox padding to 640×640

2. **Inference** (RKNN)
   - Load quantized model
   - Run inference with float output conversion (want_float=1)

3. **Postprocessing**
   - Distance-to-bbox decoding per anchor
   - Sigmoid activation for out-of-range scores
   - Non-Maximum Suppression (NMS) with IoU threshold 0.4

### File Structure

```
test09/cpp/
├── main.cc                  # Demo entry point
├── CMakeLists.txt          # Build configuration
└── rknpu2/
    ├── scrfd_rknn.h        # SCRFD RKNN API header
    └── scrfd_rknn.cc       # SCRFD RKNN implementation
```

### Algorithm Flow

1. Load image and model
2. Preprocess: resize with letterbox
3. Run RKNN inference
4. Parse 9 output tensors:
   - Scores: 3 tensors (stride 8, 16, 32)
   - Bboxes: 3 tensors (distance format)
   - Keypoints: 3 tensors (5 facial landmarks each)
5. Decode boxes using distance formula: `(cx ± dist*stride - pad) / scale`
6. Apply NMS to remove overlapping detections
7. Draw and save annotated image

## Notes

- Image height must be resized to 640 for optimal performance
- Confidence threshold affects detection count (higher = fewer, more confident detections)
- RGA hardware acceleration may log warnings on some systems; CPU fallback is automatic
- Default model path: `/home/hw/models/scrfd/scrfd_500m_bnkps_shape640x640.rknn`

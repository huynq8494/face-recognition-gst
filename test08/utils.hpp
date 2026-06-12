#pragma once
#include <opencv2/opencv.hpp>

// Reference 5 facial keypoints for alignment (112x112 standard)
static const float FACE_ALIGN_REF_POINTS[5][2] = {
    {38.2946f, 51.6963f},
    {73.5318f, 51.5014f},
    {56.0252f, 71.7366f},
    {41.5493f, 92.3655f},
    {70.7299f, 92.2041f}
};

struct FaceDetection {
    float x1, y1, x2, y2, score;
    float kps[5][2]; // 5 facial keypoints (x, y)
};

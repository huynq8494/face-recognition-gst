#ifndef _RKNN_SCRFD_H_
#define _RKNN_SCRFD_H_

#include "common.h"
#include "rknn_api.h"

#define OBJ_NUMB_MAX_SIZE 128

// Detection result types
typedef struct {
    image_rect_t box;
    float prop;
    int cls_id;
} object_detect_result;

typedef struct {
    int id;
    int count;
    object_detect_result results[OBJ_NUMB_MAX_SIZE];
} object_detect_result_list;

// SCRFD RKNN context structure
typedef struct {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
    int model_channel;
    int model_width;
    int model_height;
    bool is_quant;
} rknn_app_context_t;

int init_scrfd_model(const char* model_path, rknn_app_context_t* app_ctx);
int release_scrfd_model(rknn_app_context_t* app_ctx);
int inference_scrfd_model(rknn_app_context_t* app_ctx, image_buffer_t* img, object_detect_result_list* od_results, float conf_thresh=0.4f);

#endif //_RKNN_SCRFD_H_

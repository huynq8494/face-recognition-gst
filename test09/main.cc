#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rknpu2/scrfd_rknn.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"

int main(int argc, char **argv)
{
    const char *model_path = "/home/hw/models/scrfd/scrfd_500m_bnkps_shape640x640.rknn";
    const char *image_path = (argc <= 1) ? "/home/hw/data/images/scrfd_test.jpg" : argv[1];
    float conf_thresh = (argc <= 2) ? 0.4f : std::atof(argv[2]);

    int ret;
    rknn_app_context_t rknn_app_ctx;
    memset(&rknn_app_ctx, 0, sizeof(rknn_app_context_t));

    ret = init_scrfd_model(model_path, &rknn_app_ctx);
    if (ret != 0)
    {
        printf("init_scrfd_model fail! ret=%d model_path=%s\n", ret, model_path);
        goto out;
    }

    image_buffer_t src_image;
    memset(&src_image, 0, sizeof(image_buffer_t));
    ret = read_image(image_path, &src_image);
    if (ret != 0)
    {
        printf("read image fail! ret=%d image_path=%s\n", ret, image_path);
        goto out;
    }

    object_detect_result_list od_results;

    ret = inference_scrfd_model(&rknn_app_ctx, &src_image, &od_results, conf_thresh);
    if (ret != 0)
    {
        printf("inference_scrfd_model fail! ret=%d\n", ret);
        goto out;
    }

    // Draw boxes and confidence scores
    char text[256];
    for (int i = 0; i < od_results.count; i++)
    {
        object_detect_result *det_result = &(od_results.results[i]);
         printf("face @ (%d %d %d %d) %.3f\n",
             det_result->box.left, det_result->box.top,
             det_result->box.right, det_result->box.bottom,
             det_result->prop);
        int x1 = det_result->box.left;
        int y1 = det_result->box.top;
        int x2 = det_result->box.right;
        int y2 = det_result->box.bottom;

        draw_rectangle(&src_image, x1, y1, x2 - x1, y2 - y1, COLOR_BLUE, 3);

        sprintf(text, "face %.1f%%", det_result->prop * 100);
        draw_text(&src_image, text, x1, y1 - 20, COLOR_RED, 10);
    }

    write_image("out.png", &src_image);

out:
    ret = release_scrfd_model(&rknn_app_ctx);
    if (ret != 0)
    {
        printf("release_scrfd_model fail! ret=%d\n", ret);
    }

    if (src_image.virt_addr != NULL)
    {
        free(src_image.virt_addr);
    }

    return 0;
}

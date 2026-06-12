// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rknpu2/scrfd_rknn.h"
#include "image_utils.h"
#include "file_utils.h"
#include "image_drawing.h"


/*-------------------------------------------
              SCRFD Face Detection Demo
-------------------------------------------*/
int main(int argc, char **argv)
{
    // Usage:
    // 1) ./test09 <image_path>
    //    -> Uses default SCRFD model: /home/huynq/models/scrfd/scrfd_500m_bnkps_shape640x640.rknn
    // 2) ./test09 <image_path> <conf_threshold>
    //    -> Uses default SCRFD model with custom confidence (0.0-1.0)
    // 3) ./test09 <model_path> <image_path> [conf_threshold]
    //    -> Uses specified RKNN model
    const char *default_model_path = "/home/huynq/models/scrfd/scrfd_500m_bnkps_shape640x640.rknn";

    const char *model_path = NULL;
    const char *image_path = NULL;

    auto is_number = [](const char *s)->bool{
        if (!s || !*s) return false;
        char *endptr = NULL;
        strtod(s, &endptr);
        return endptr && *endptr == '\0';
    };

    float conf_thresh = 0.4f;

    if (argc == 2) {
        // only image provided -> use default model
        model_path = default_model_path;
        image_path = argv[1];
    } else if (argc == 3) {
        // could be: <model> <image> OR <image> <conf>
        if (is_number(argv[2])) {
            model_path = default_model_path;
            image_path = argv[1];
            conf_thresh = (float)atof(argv[2]);
        } else {
            model_path = argv[1];
            image_path = argv[2];
        }
    } else if (argc == 4) {
        model_path = argv[1];
        image_path = argv[2];
        if (is_number(argv[3])) conf_thresh = (float)atof(argv[3]);
    } else {
        printf("SCRFD Face Detection Demo\n");
        printf("Usage:\n");
        printf("  %s <image_path>\n", argv[0]);
        printf("  %s <image_path> <conf_threshold>\n", argv[0]);
        printf("  %s <model_path> <image_path> [conf_threshold]\n", argv[0]);
        printf("Default model: %s\n", default_model_path);
        return -1;
    }
    

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

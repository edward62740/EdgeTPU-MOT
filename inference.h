#pragma once

#include <map>
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
#include "main.h"

namespace coralmicro
{
    namespace inference
    {

        extern int img_width;
        extern int img_height;
        extern unsigned int bbox_buf_size;
        extern char *bbox_buf;
        extern std::vector<uint8_t> *img_copy;
        extern std::string bbox_string;

        [[noreturn]] void inferenceTask(void *param);
        bool CalculateAnchorBox(unsigned int idx, float *anchor);
        float CalculateIOU(shared::BBox *bbox1, shared::BBox *bbox2);

        // Function to get COCO class labels
        std::map<int, std::string> _getCocoLabels();

    }
}

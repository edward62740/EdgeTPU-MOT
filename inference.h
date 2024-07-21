#pragma once

#include "main.h"
#include <map>
#include <cstdio>
#include <vector>
#include <cmath>
#include <string>

namespace coralmicro
{
    namespace inference
    {

        extern int img_width;
        extern int img_height;
        extern unsigned int bbox_buf_size;
        extern char *bbox_buf;
        extern std::vector<uint8_t> *img_copy;

        [[noreturn]] void inferenceTask(void *param);
        bool CalculateAnchorBox(unsigned int idx, float *anchor);
        float CalculateIOU(shared::BBox *bbox1, shared::BBox *bbox2);

        // Function to get COCO class labels
        std::map<int, std::string> _getCocoLabels();

    }
}

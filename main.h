#pragma once
#include "third_party/freertos_kernel/include/semphr.h"


namespace shared
{
    extern SemaphoreHandle_t img_mutex;
    extern SemaphoreHandle_t bbox_mutex;

    // Image result struct
    typedef struct
    {
        std::string info;
        std::vector<uint8_t> *jpeg;
    } ImgResult;

    // Bounding box struct
    typedef struct
    {
        float id;
        float score;
        float ymin;
        float xmin;
        float ymax;
        float xmax;
    } BBox;

    extern bool isMotionDetected;

}

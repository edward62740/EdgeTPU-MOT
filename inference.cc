#include <cstdio>
#include "inference.h"

#include "libs/base/led.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"


void inference(void * pv)
{
    LedSet(coralmicro::Led::kUser, true);
    vTaskDelay(50);
    LedSet(coralmicro::Led::kUser, false);
    vTaskDelay(50);
}
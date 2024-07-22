#include <cstdio>
#include <vector>
#include <cmath>

#include "libs/base/http_server.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"
#include "libs/base/utils.h"
#include "libs/base/gpio.h"
#include "libs/camera/camera.h"
#include "libs/libjpeg/jpeg.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/freertos_kernel/include/semphr.h"
#include "libs/base/wifi.h"
#include "third_party/freertos_kernel/include/timers.h"

#include "inference.h"
#include "stream.h"

bool shared::isMotionDetected = false;
static constexpr size_t max_bboxes = 5;
unsigned int coralmicro::inference::bbox_buf_size = 100 + (max_bboxes * 200) + 1; // Assuming max_bboxes is properly declared
char *coralmicro::inference::bbox_buf = (char *)malloc(bbox_buf_size * sizeof(char));

int coralmicro::inference::img_width = 0;
int coralmicro::inference::img_height = 0;
SemaphoreHandle_t shared::img_mutex = xSemaphoreCreateMutex();
SemaphoreHandle_t shared::bbox_mutex = xSemaphoreCreateMutex();
coralmicro::HttpServer http_server;
namespace coralmicro
{

  // Globals
  constexpr char kIndexFileName[] = "/index.html";
  constexpr char kCameraStreamUrlPrefix[] = "/camera_stream";
  constexpr char kBoundingBoxPrefix[] = "/bboxes";
// Copy of image data for HTTP server





  HttpServer::Content UriHandler(const char *uri)
  {

    // Give client main page
    if (StrEndsWith(uri, "index.shtml") ||
        StrEndsWith(uri, "coral_micro_camera.html"))
    {
      return std::string(kIndexFileName);

      // Give client compressed image data
    }
    else if (StrEndsWith(uri, kCameraStreamUrlPrefix))
    {

      // Read image from shared memory and compress to JPG
      std::vector<uint8_t> jpeg;
      if (xSemaphoreTake(shared::img_mutex, portMAX_DELAY) == pdTRUE)
      {
        JpegCompressRgb(
            inference::img_copy->data(),
            inference::img_width,
            inference::img_height,
            30, // Quality
            &jpeg);
        xSemaphoreGive(shared::img_mutex);
      }

      return jpeg;

      // Give client bounding box info
    }
    else if (StrEndsWith(uri, kBoundingBoxPrefix))
    {

      // Read bounding box info from shared memory and convert to vector of bytes
      char bbox_info_copy[inference::bbox_buf_size];
      std::vector<uint8_t> bbox_info_bytes;
      if (xSemaphoreTake(shared::bbox_mutex, portMAX_DELAY) == pdTRUE)
      {
        std::strcpy(bbox_info_copy, inference::bbox_buf);
        bbox_info_bytes.assign(
            bbox_info_copy,
            bbox_info_copy + std::strlen(bbox_info_copy));
        xSemaphoreGive(shared::bbox_mutex);
      }

      // TODO: Figure out the multi-request or race condition bug that is causing
      // the bbox_info_bytes to be corrupted. The workaround is to have the
      // client timeout if it doesn't get a response in some amount of time.

      return bbox_info_bytes;
    }

    return {};
  }


  /*******************************************************************************
   * Main
   */

  void Main()
  {

    shared::isMotionDetected = false;

    // Initialize image mutex

    if (shared::img_mutex == NULL)
    {
      printf("Error creating image mutex\r\n");

    }

    // Initialize bounding box mutex

    if (shared::bbox_mutex == NULL)
    {
      printf("Error creating bbox mutex\r\n");

    }
    CameraMotionDetectionConfig config{};
    CameraTask::GetSingleton()->GetMotionDetectionConfigDefault(config);
    config.cb = [](void *param)
    {
      shared::isMotionDetected = true;
    };

    // Initialize camera
    CameraTask::GetSingleton()->SetPower(true);
    CameraTask::GetSingleton()->SetMotionDetectionConfig(config);
    CameraTask::GetSingleton()->Enable(CameraMode::kStreaming);


    printf("Starting inference task\r\n");

    xTaskCreate(
        &inference::inferenceTask,
        "InferenceTask",
        configMINIMAL_STACK_SIZE * 30,
        nullptr,
        kAppTaskPriority - 1, // Make inference lower than console/server
        nullptr);
    xTaskCreate(
        &stream::streamTask,
        "StreamTask",
        configMINIMAL_STACK_SIZE * 30,
        nullptr,
        kAppTaskPriority - 2, // Make inference lower than console/server
        nullptr);

    if (!WiFiTurnOn(/*default_iface=*/true))
    {
      printf("Unable to bring up WiFi...\r\n");
      vTaskSuspend(nullptr);
    }
    WiFiSetDefaultSsid("");
    WiFiSetDefaultPsk("");
    if (!WiFiConnect(10))
    {
      printf("Unable to connect to WiFi...\r\n");
      vTaskSuspend(nullptr);
    }
    if (auto wifi_ip = WiFiGetIp())
    {
      printf("Serving on: %s\r\n", wifi_ip->c_str());
    }
    else
    {
      printf("Failed to get Wifi Ip\r\n");
      vTaskSuspend(nullptr);
    }

    // Initialize HTTP server (attach request handler)

    http_server.AddUriHandler(UriHandler);
    UseHttpServer(&http_server);


    // Main will go to sleep
    vTaskSuspend(nullptr);
  }
} // namespace coralmicro

/**
 * Entrypoint
 */
extern "C" void app_main(void *param)
{
  (void)param;

  coralmicro::GpioSetMode(coralmicro::Gpio::kCameraPrivacyOverride, coralmicro::GpioMode::kOutput); // disable cam led
  // coralmicro::GpioSetMode(coralmicro::Gpio::kUserLed, coralmicro::GpioMode::kInput); // z state tpu led
  coralmicro::GpioSet(coralmicro::Gpio::kCameraPrivacyOverride, false);
  coralmicro::Main();
}

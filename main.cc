#include <cstring>
#include <vector>

#include "libs/base/filesystem.h"
#include "libs/base/gpio.h"
#include "libs/base/led.h"
#include "libs/camera/camera.h"
#include "libs/rpc/rpc_http_server.h"
#include "libs/tensorflow/detection.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/mjson/src/mjson.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"


namespace coralmicro {
namespace {
constexpr char kModelPath[] =
    "/models/tf2_ssd_mobilenet_v2_coco17_ptq_edgetpu.tflite";
// An area of memory to use for input, output, and intermediate arrays.
constexpr int kTensorArenaSize = 8 * 1024 * 1024;
STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);

bool DetectFromCamera(tflite::MicroInterpreter* interpreter, int model_width,
                      int model_height,
                      std::vector<tensorflow::Object>* results,
                      std::vector<uint8>* image) {
  CHECK(results != nullptr);
  CHECK(image != nullptr);
  auto* input_tensor = interpreter->input_tensor(0);
  CameraFrameFormat fmt{CameraFormat::kRgb,   CameraFilterMethod::kBilinear,
                        CameraRotation::k90, model_width,
                        model_height,         false,
                        image->data()};

  CameraTask::GetSingleton()->Trigger();
  if (!CameraTask::GetSingleton()->GetFrame({fmt})) return false;

  std::memcpy(tflite::GetTensorData<uint8_t>(input_tensor), image->data(),
              image->size());
  if (interpreter->Invoke() != kTfLiteOk) return false;

  *results = tensorflow::GetDetectionResults(interpreter, 0.5, 1);
  return true;
}

void DetectRpc(struct jsonrpc_request* r) {
  auto* interpreter =
      static_cast<tflite::MicroInterpreter*>(r->ctx->response_cb_data);
  auto* input_tensor = interpreter->input_tensor(0);
  int model_height = input_tensor->dims->data[1];
  int model_width = input_tensor->dims->data[2];
  std::vector<uint8> image(model_height * model_width *
                           CameraFormatBpp(CameraFormat::kRgb));
  std::vector<tensorflow::Object> results;
  if (DetectFromCamera(interpreter, model_width, model_height, &results,
                       &image)) {
    if (!results.empty()) {
      const auto& result = results[0];
      jsonrpc_return_success(
          r,
          "{%Q: %d, %Q: %d, %Q: %V, %Q: {%Q: %d, %Q: %g, %Q: %g, %Q: %g, "
          "%Q: %g, %Q: %g}}",
          "width", model_width, "height", model_height, "base64_data",
          image.size(), image.data(), "detection", "id", result.id, "score",
          result.score, "xmin", result.bbox.xmin, "xmax", result.bbox.xmax,
          "ymin", result.bbox.ymin, "ymax", result.bbox.ymax);
      return;
    }
    jsonrpc_return_success(r, "{%Q: %d, %Q: %d, %Q: %V, %Q: None}", "width",
                           model_width, "height", model_height, "base64_data",
                           image.size(), image.data(), "detection");
    return;
  }
  jsonrpc_return_error(r, -1, "Failed to detect image from camera.", nullptr);
}

void DetectConsole(tflite::MicroInterpreter* interpreter) {
  auto* input_tensor = interpreter->input_tensor(0);
  int model_height = input_tensor->dims->data[1];
  int model_width = input_tensor->dims->data[2];
  std::vector<uint8> image(model_height * model_width *
                           CameraFormatBpp(CameraFormat::kRgb));
  std::vector<tensorflow::Object> results;
  if (DetectFromCamera(interpreter, model_width, model_height, &results,
                       &image)) {
    printf("%s\r\n", tensorflow::FormatDetectionOutput(results).c_str());
  } else {
    printf("Failed to detect image from camera.\r\n");
  }
}

[[noreturn]] void Main() {
  printf("Detection Camera Example!\r\n");
  // Turn on Status LED to show the board is on.
  LedSet(Led::kStatus, true);

  std::vector<uint8_t> model;
  if (!LfsReadFile(kModelPath, &model)) {
    printf("ERROR: Failed to load %s\r\n", kModelPath);
    vTaskSuspend(nullptr);
  }

  auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice();
  if (!tpu_context) {
    printf("ERROR: Failed to get EdgeTpu context\r\n");
    vTaskSuspend(nullptr);
  }

  tflite::MicroErrorReporter error_reporter;
  tflite::MicroMutableOpResolver<3> resolver;
  resolver.AddDequantize();
  resolver.AddDetectionPostprocess();
  resolver.AddCustom(kCustomOp, RegisterCustomOp());

  tflite::MicroInterpreter interpreter(tflite::GetModel(model.data()), resolver,
                                       tensor_arena, kTensorArenaSize,
                                       &error_reporter);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    printf("ERROR: AllocateTensors() failed\r\n");
    vTaskSuspend(nullptr);
  }

  if (interpreter.inputs().size() != 1) {
    printf("ERROR: Model must have only one input tensor\r\n");
    vTaskSuspend(nullptr);
  }

  // Starting Camera.
  CameraTask::GetSingleton()->SetPower(true);
  CameraTask::GetSingleton()->Enable(CameraMode::kTrigger);

  printf("Initializing detection server...\r\n");
  jsonrpc_init(nullptr, &interpreter);
  jsonrpc_export("detect_from_camera", DetectRpc);
  UseHttpServer(new JsonRpcHttpServer);
  printf("Detection server ready!\r\n");
  GpioConfigureInterrupt(
      Gpio::kUserButton, GpioInterruptMode::kIntModeFalling,
      [handle = xTaskGetCurrentTaskHandle()]() { xTaskResumeFromISR(handle); },
      /*debounce_interval_us=*/50 * 1e3);
  while (true) {
    vTaskSuspend(nullptr);
    DetectConsole(&interpreter);
  }
}

}  // namespace
}  // namespace coralmicro

extern "C" void app_main(void* param) {
  (void)param;
  coralmicro::Main();
}

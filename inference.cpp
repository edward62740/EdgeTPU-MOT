#include <cstdio>
#include <vector>
#include <cmath>
#include <string>

#include "libs/base/filesystem.h"
#include "libs/base/led.h"
#include "libs/base/strings.h"

#include "libs/base/gpio.h"
#include "libs/camera/camera.h"
#include "libs/tensorflow/detection.h"
#include "libs/tensorflow/utils.h"
#include "libs/tpu/edgetpu_manager.h"
#include "libs/tpu/edgetpu_op.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "third_party/freertos_kernel/include/semphr.h"
#include "third_party/freertos_kernel/include/timers.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_error_reporter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_interpreter.h"
#include "third_party/tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "metadata.hpp"

#include "inference.h"

namespace coralmicro
{

    namespace inference
    {
        std::vector<uint8_t> *img_copy = nullptr;

        constexpr char kModelPath[] =
            "/ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite";
        constexpr int kTensorArenaSize = 8 * 1024 * 1024;
        STATIC_TENSOR_ARENA_IN_SDRAM(tensor_arena, kTensorArenaSize);
        static std::vector<uint8_t> *img_ptr;
        void ProcessOutput(const float *output, size_t d,
                           std::vector<float> &bboxes, std::vector<float> &ids,
                           std::vector<float> &scores, size_t &count, size_t num_classes)
        {
            for (size_t i = 0; i < d; ++i)
            {
                // Bounding box coordinates
                float x = output[0];
                float y = output[1];
                float w = output[2];
                float h = output[3];

                // Append bounding box coordinates
                bboxes.push_back(x - w / 2); // ymin
                bboxes.push_back(y - h / 2); // xmin
                bboxes.push_back(x + w / 2); // ymax
                bboxes.push_back(y + h / 2); // xmax

                // Find the class with the maximum score
                float max_score = 0;
                int max_class = -1;
                for (size_t j = 0; j < num_classes; ++j)
                {
                    float score = output[4 + j];
                    if (score > max_score)
                    {
                        max_score = score;

                        max_class = static_cast<int>(j);
                    }
                }

                // Append scores and class IDs
                scores.push_back(max_score);
                ids.push_back(static_cast<float>(max_class));
                ++count;

                // Move the output pointer to the next set of predictions
                output += d;
            }
        }
        /**
         * Loop forever taking images from the camera and performing inference
         */
        [[noreturn]] void inferenceTask(void *param)
        {

            // Used for calculating FPS
            unsigned long dtime;
            unsigned long timestamp;
            unsigned long timestamp_prev = xTaskGetTickCount() *
                                           (1000 / configTICK_RATE_HZ);

            // Load model
            std::vector<uint8_t> model;
            if (!LfsReadFile(kModelPath, &model))
            {
                printf("ERROR: Failed to load %s\r\n", kModelPath);
                vTaskSuspend(nullptr);
            }

            // Initialize TPU
            auto tpu_context = EdgeTpuManager::GetSingleton()->OpenDevice(/*coralmicro::PerformanceMode::kLow*/);
            if (!tpu_context)
            {
                printf("ERROR: Failed to get EdgeTpu context\r\n");
                vTaskSuspend(nullptr);
            }

            // Initialize ops
            tflite::MicroErrorReporter error_reporter;
            tflite::MicroMutableOpResolver<3> resolver;
            resolver.AddDequantize();
            resolver.AddDetectionPostprocess();

            // resolver.AddTranspose();
            // resolver.AddReshape();
            // resolver.AddConcatenation();
            // resolver.AddStridedSlice();
            // resolver.AddLogistic();
            // resolver.AddQuantize();

            resolver.AddCustom(kCustomOp, RegisterCustomOp());

            // Initialize TFLM interpreter for inference
            tflite::MicroInterpreter interpreter(
                tflite::GetModel(model.data()),
                resolver,
                tensor_arena,
                kTensorArenaSize,
                &error_reporter);
            if (interpreter.AllocateTensors() != kTfLiteOk)
            {
                printf("ERROR: AllocateTensors() failed\r\n");
                vTaskSuspend(nullptr);
            }

            // Check model input tensor size
            if (interpreter.inputs().size() != 1)
            {
                printf("ERROR: Model must have only one input tensor\r\n");
                vTaskSuspend(nullptr);
            }
            LedSet(coralmicro::Led::kTpu, false);

            // Configure model inputs and outputs
            auto *input_tensor = interpreter.input_tensor(0);
            img_height = input_tensor->dims->data[1];
            img_width = input_tensor->dims->data[2];
            img_ptr = new std::vector<uint8>(img_height * img_width *
                                             CameraFormatBpp(CameraFormat::kRgb));

            std::vector<tensorflow::Object> results;

            printf("Input size is %d by %d", img_width, img_height);
            // Copy image to separate buffer for HTTP server
            img_copy = new std::vector<uint8_t>(img_ptr->size());
            for (int j = 0; j < 1; j++)
            {
                TfLiteTensor *tensor = interpreter.output_tensor(j);
                printf("Tensor %d dims: ", j);
                for (int i = 0; i < tensor->dims->size; ++i)
                {

                    printf("%d ", tensor->dims->data[i]);
                }
                printf("\r\n");
            }
            printf("Starting inference\r\n");
            // Do forever
            while (true)
            {

                std::vector<std::vector<float>> bbox_list;

                // Calculate time between inferences
                timestamp = xTaskGetTickCount() * (1000 / configTICK_RATE_HZ);
                dtime = timestamp - timestamp_prev;
                timestamp_prev = timestamp;

                LedSet(coralmicro::Led::kTpu, false);
                // Get frame from camera using the configuration we set (~38 ms)
                if (xSemaphoreTake(shared::img_mutex, portMAX_DELAY) == pdTRUE)
                {

                    // Configure camera image
                    CameraFrameFormat fmt{
                        CameraFormat::kRgb,
                        CameraFilterMethod::kBilinear,
                        CameraRotation::k270,
                        img_height,
                        img_width,
                        false,           // Preserve ratio
                        img_ptr->data(), // Where the image is saved
                        true             // Auto white balance
                    };

                    // Take a photo
                    if (!CameraTask::GetSingleton()->GetFrame({fmt}))
                    {
                        printf("ERROR: Could not capture frame from camera\r\n");
                        continue;
                    }

                    // Turn status LED on to let the user know we're taking a photo
                    LedSet(Led::kUser, true);

                    // Copy image to input tensor (~6 ms)
                    std::memcpy(
                        tflite::GetTensorData<uint8_t>(input_tensor),
                        img_ptr->data(),
                        img_ptr->size());

                    LedSet(Led::kUser, false);

                    if (interpreter.Invoke() != kTfLiteOk)
                    {
                        printf("ERROR: Inference failed\r\n");
                    }
      
                    printf("Model running\r\n");
                    /*
                    float *output = tflite::GetTensorData<float>(interpreter.output_tensor(0));

                    size_t num_classes = 80; // Number of classes

                    // Print the tensor data
                    // Process the output
                    size_t output_size = 756;
                    // size_t num_classes = 80; // For COCO dataset
                    std::vector<float> bboxes, ids, scores;
                    size_t count = 0;
                    ProcessOutput(output, output_size, bboxes, ids, scores, count, num_classes);

                    // Call the GetDetectionResults function
                    float threshold = 0.5f;
                    size_t top_k = 10;
                    results = tensorflow::GetDetectionResults(bboxes.data(), ids.data(), scores.data(), count, threshold, top_k);
                    for (const auto &obj : results)
                    {
                        printf("Object: id=%d, score=%.2f, bbox=[%.2f, %.2f, %.2f, %.2f]\r\n", obj.id, obj.score, obj.bbox.ymin, obj.bbox.xmin, obj.bbox.ymax, obj.bbox.xmax);
                    }
                    */

                    results = tensorflow::GetDetectionResults(&interpreter, 0.6, 10);

                    LedSet(coralmicro::Led::kTpu, false);
                    if (!results.empty())
                    {
                        for (const auto &result : results)
                        {
                            float x_min = result.bbox.xmin;
                            float x_max = result.bbox.xmax;
                            float y_min = result.bbox.ymin;
                            float y_max = result.bbox.ymax;
                            float score = result.score;
                            int c = result.id;

                            bbox_list.push_back({(float)c, score,
                                                 y_min, x_min, y_max, x_max});
                        }
                    }

                    std::memcpy(
                        img_copy->data(),
                        img_ptr->data(),
                        img_ptr->size());

                    // Unlock critical section
                    xSemaphoreGive(shared::img_mutex);
                }

                // Sort bboxes by score (descending order)
                std::sort(bbox_list.begin(), bbox_list.end(),
                          [](const std::vector<float> &a, const std::vector<float> &b)
                          {
                              return a[1] > b[1];
                          });

                // Perform NMS
                for (uint32_t i = 0; i < bbox_list.size(); ++i)
                {
                    for (uint32_t j = i + 1; j < bbox_list.size(); ++j)
                    {
                        shared::BBox bbox1 = {bbox_list[i][0], bbox_list[i][1], bbox_list[i][2],
                                              bbox_list[i][3], bbox_list[i][4], bbox_list[i][5]};
                        shared::BBox bbox2 = {bbox_list[j][0], bbox_list[j][1], bbox_list[j][2],
                                              bbox_list[j][3], bbox_list[j][4], bbox_list[j][5]};

                        float iou = CalculateIOU(&bbox1, &bbox2);
                        if (iou > 0.2)
                        {
                            // Erase bbox_list[j] if IoU is higher than threshold
                            bbox_list.erase(bbox_list.begin() + j);
                            printf("Erased");
                            --j; // Adjust j to account for the erased element
                        }
                    }
                }

                // Determine number of bboxes to send
                size_t num_bboxes_output = bbox_list.size();
                // Convert top k bboxes to JSON string
                std::string bbox_string = "{\"dtime\": " + std::to_string(dtime) + ",";
                bbox_string += "\"isMotionDetected\": " + std::to_string(shared::isMotionDetected) + ",";
                bbox_string += "\"bboxes\":[";
                std::map<int, std::string> coco_labels = _getCocoLabels();
                for (unsigned int i = 0; i < num_bboxes_output; ++i)
                {
                    int class_id = static_cast<int>(bbox_list[i][0]);
                    std::string class_label = coco_labels[class_id + 1];
                    bbox_string += "{\"id\":\"" + std::to_string(class_id) + "-" + class_label;
                    if (shared::isMotionDetected && class_id == 0)
                        bbox_string += "   TRACKING";
                    bbox_string += "\",";
                    bbox_string += "\"score\":" + std::to_string(bbox_list[i][1]) + ",";
                    bbox_string += "\"xmin\":" + std::to_string(bbox_list[i][3]) + ",";
                    bbox_string += "\"ymin\":" + std::to_string(bbox_list[i][2]) + ",";
                    bbox_string += "\"xmax\":" + std::to_string(bbox_list[i][5]) + ",";
                    bbox_string += "\"ymax\":" + std::to_string(bbox_list[i][4]) + "}";
                    if (i != num_bboxes_output - 1)
                    {
                        bbox_string += ",";
                    }
                }
                bbox_string += "]}";
                shared::isMotionDetected = false;

                // Check length of JSON string
                if (bbox_string.length() > bbox_buf_size)
                {
                    printf("ERROR: Bounding box JSON string too long\r\n");
                    continue;
                }

                // Convert global char array
                if (xSemaphoreTake(shared::bbox_mutex, portMAX_DELAY) == pdTRUE)
                {
                    std::strcpy(bbox_buf, bbox_string.c_str());
                    xSemaphoreGive(shared::bbox_mutex);
                }

                // Print bounding box JSON string
                printf("%s\r\n", bbox_buf);

                // Sleep to let other tasks run
                vTaskDelay(pdMS_TO_TICKS(1));
            }
        }

        /**
         * Calculate anchor box coordinates based on index and metadata
         */
        bool CalculateAnchorBox(unsigned int idx, float *anchor)
        {

            unsigned int sector = 0;
            float x_idx;
            float x_center;
            float y_idx;
            float y_center;
            float w;
            float h;

            // Check index
            if (idx >= metadata::num_anchors)
            {
                return false;
            }

            // Find the sector that the index belongs in
            for (unsigned int s = 0; s < metadata::num_sectors; ++s)
            {
                if (idx >= metadata::reset_idxs[s])
                {
                    sector = s;
                }
            }

            // Find the X centert
            x_idx = (idx % metadata::num_xs_per_y[sector]) /
                    metadata::num_anchors_per_coord;
            x_center = (metadata::x_strides[sector] / 2.0f) +
                       (x_idx * metadata::x_strides[sector]);

            // Find the Y center
            y_idx = (idx - metadata::reset_idxs[sector]) /
                    metadata::num_xs_per_y[sector];
            y_center = (metadata::y_strides[sector] / 2.0f) +
                       (y_idx * metadata::y_strides[sector]);

            // Find the width and height
            w = metadata::widths[sector][idx % metadata::num_anchors_per_coord];
            h = metadata::heights[sector][idx % metadata::num_anchors_per_coord];

            // Save anchor box coordinates
            anchor[0] = x_center;
            anchor[1] = y_center;
            anchor[2] = w;
            anchor[3] = h;

            return true;
        }

        /**
         * Calculate intersection over union (IOU) between two bounding boxes
         */
        float CalculateIOU(shared::BBox *bbox1, shared::BBox *bbox2)
        {

            // Calculate intersection
            float x_min = std::max(bbox1->xmin, bbox2->xmin);
            float y_min = std::max(bbox1->ymin, bbox2->ymin);
            float x_max = std::min(bbox1->xmax, bbox2->xmax);
            float y_max = std::min(bbox1->ymax, bbox2->ymax);
            float intersection = std::max(0.0f, x_max - x_min) *
                                 std::max(0.0f, y_max - y_min);

            // Calculate union
            float bbox1_area = (bbox1->xmax - bbox1->xmin) *
                               (bbox1->ymax - bbox1->ymin);
            float bbox2_area = (bbox2->xmax - bbox2->xmin) *
                               (bbox2->ymax - bbox2->ymin);
            float union_area = bbox1_area + bbox2_area - intersection;

            // Calculate IOU
            float iou = 0.0f;
            if (union_area > 0.0f)
            {
                iou = intersection / union_area;
            }

            return iou;
        }

        // Function to get COCO class labels
        std::map<int, std::string> _getCocoLabels()
        {
            return {
                {0, "__BACKGROUND__"},
                {1, "PERSON"},
                {2, "BICYCLE"},
                {3, "CAR"},
                {4, "MOTORCYCLE"},
                {5, "AIRPLANE"},
                {6, "BUS"},
                {7, "TRAIN"},
                {8, "TRUCK"},
                {9, "BOAT"},
                {10, "TRAFFIC LIGHT"},
                {11, "FIRE HYDRANT"},
                {12, "N/A"}, // Placeholder for index 12
                {13, "STOP SIGN"},
                {14, "PARKING METER"},
                {15, "BENCH"},
                {16, "BIRD"},
                {17, "CAT"},
                {18, "DOG"},
                {19, "HORSE"},
                {20, "SHEEP"},
                {21, "COW"},
                {22, "ELEPHANT"},
                {23, "BEAR"},
                {24, "ZEBRA"},
                {25, "GIRAFFE"},
                {26, "N/A"}, // Placeholder for index 26
                {27, "BACKPACK"},
                {28, "UMBRELLA"},
                {29, "N/A"}, // Placeholder for index 29
                {30, "N/A"}, // Placeholder for index 30
                {31, "HANDBAG"},
                {32, "TIE"},
                {33, "SUITCASE"},
                {34, "FRISBEE"},
                {35, "SKIS"},
                {36, "SNOWBOARD"},
                {37, "SPORTS BALL"},
                {38, "KITE"},
                {39, "BASEBALL BAT"},
                {40, "BASEBALL GLOVE"},
                {41, "SKATEBOARD"},
                {42, "SURFBOARD"},
                {43, "TENNIS RACKET"},
                {44, "BOTTLE"},
                {45, "N/A"}, // Placeholder for index 45
                {46, "WINE GLASS"},
                {47, "CUP"},
                {48, "FORK"},
                {49, "KNIFE"},
                {50, "SPOON"},
                {51, "BOWL"},
                {52, "BANANA"},
                {53, "APPLE"},
                {54, "SANDWICH"},
                {55, "ORANGE"},
                {56, "BROCCOLI"},
                {57, "CARROT"},
                {58, "HOT DOG"},
                {59, "PIZZA"},
                {60, "DONUT"},
                {61, "CAKE"},
                {62, "CHAIR"},
                {63, "COUCH"},
                {64, "POTTED PLANT"},
                {65, "BED"},
                {66, "N/A"}, // Placeholder for index 66
                {67, "DINING TABLE"},
                {68, "N/A"}, // Placeholder for index 68
                {69, "N/A"}, // Placeholder for index 69
                {70, "TOILET"},
                {71, "N/A"}, // Placeholder for index 71
                {72, "TV"},
                {73, "LAPTOP"},
                {74, "MOUSE"},
                {75, "REMOTE"},
                {76, "KEYBOARD"},
                {77, "CELL PHONE"},
                {78, "MICROWAVE"},
                {79, "OVEN"},
                {80, "TOASTER"},
                {81, "SINK"},
                {82, "REFRIGERATOR"},
                {83, "N/A"}, // Placeholder for index 83
                {84, "BOOK"},
                {85, "CLOCK"},
                {86, "VASE"},
                {87, "SCISSORS"},
                {88, "TEDDY BEAR"},
                {89, "HAIR DRIER"},
                {90, "TOOTHBRUSH"}};
        }

    }
}
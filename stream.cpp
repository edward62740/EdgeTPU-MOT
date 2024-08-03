#include "third_party/nxp/rt1176-sdk/middleware/lwip/src/include/lwip/sockets.h"
#include "third_party/freertos_kernel/include/FreeRTOS.h"
#include "third_party/freertos_kernel/include/task.h"
#include "libs/libjpeg/jpeg.h"

#include "inference.h"
#include "stream.h"

namespace coralmicro::stream
{
    [[noreturn]] void streamTask(void *param)
    {
        (void)param;
        int listening_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        struct timeval tv
        {
        };
        tv.tv_sec = 2;
        tv.tv_usec = 0;
        setsockopt(listening_socket, 0, SO_RCVTIMEO, &tv, sizeof(tv));
        struct sockaddr_in bind_address
        {
        };
        bind_address.sin_family = AF_INET;
        bind_address.sin_port = PP_HTONS(31337);
        bind_address.sin_addr.s_addr = PP_HTONL(INADDR_ANY);

        bind(listening_socket, reinterpret_cast<struct sockaddr *>(&bind_address),
             sizeof(bind_address));

        while (true)
        {
            listen(listening_socket, 1);
            int accepted_socket = accept(listening_socket, nullptr, nullptr);
            std::vector<uint8_t> jpeg;
            const std::string delimiter = "<END_OF_IMAGE>";
            const std::string sop = "<START_OF_PAYLOAD>";
            const std::string eop = "<END_OF_PAYLOAD>";

            if (xSemaphoreTake(shared::img_mutex, portMAX_DELAY) == pdTRUE)
            {
                JpegCompressRgb(
                    inference::img_copy->data(),
                    inference::img_width,
                    inference::img_height,
                    30,
                    &jpeg);
                xSemaphoreGive(shared::img_mutex);
                std::string bbox_string;
                if (xSemaphoreTake(shared::bbox_mutex, portMAX_DELAY) == pdTRUE)
                {
      
                    bbox_string = inference::bbox_buf;
                
                    xSemaphoreGive(shared::bbox_mutex);
                }

                if (!jpeg.empty())
                {
                    // Convert char* to std::string
                    printf("Delimiter: %s\r\n", std::string(delimiter.begin(), delimiter.end()).c_str());
                    printf("BBox String: %s\r\n", bbox_string.c_str());
                    printf("EOP: %s\r\n", std::string(eop.begin(), eop.end()).c_str());

                    // Create a buffer to hold JPEG data + delimiter + bbox_string + EOP
                    std::vector<uint8_t> send_buffer;
                    send_buffer.reserve(sop.size() + jpeg.size() + delimiter.size() + bbox_string.size() + eop.size());
                    send_buffer.insert(send_buffer.end(), sop.begin(), sop.end());
                    send_buffer.insert(send_buffer.end(), jpeg.begin(), jpeg.end());
                    send_buffer.insert(send_buffer.end(), delimiter.begin(), delimiter.end());
                    send_buffer.insert(send_buffer.end(), bbox_string.begin(), bbox_string.end());
                    send_buffer.insert(send_buffer.end(), eop.begin(), eop.end());

                    int sent = send(accepted_socket, send_buffer.data(), send_buffer.size(), 0);
                    printf("Sent: %d, expected: %d\r\n", sent, send_buffer.size());
                    if (sent < 0)
                    {
                        // Handle send error
                        perror("send failed");
                    }
                }
            }
            closesocket(accepted_socket);
            vTaskDelay(pdMS_TO_TICKS(1));
        }
    }
}

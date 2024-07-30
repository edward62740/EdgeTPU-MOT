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

        int udp_socket = socket(AF_INET, SOCK_DGRAM, 0);

        struct sockaddr_in bind_address
        {
        };
        bind_address.sin_family = AF_INET;
        bind_address.sin_port = PP_HTONS(31337);
        bind_address.sin_addr.s_addr = PP_HTONL(INADDR_ANY);

        bind(udp_socket, reinterpret_cast<struct sockaddr *>(&bind_address), sizeof(bind_address));

        const char *fixed_str = "Hello socket.\r\n";
        struct sockaddr_in dest_address
        {
        };
        dest_address.sin_family = AF_INET;
        dest_address.sin_port = PP_HTONS(31337);
        dest_address.sin_addr.s_addr = inet_addr("10.10.1.102"); // Example IP address to send to
        vTaskDelay(pdMS_TO_TICKS(2000));
        while (true)
        {

            
            std::vector<uint8_t> jpeg;

            if (xSemaphoreTake(shared::img_mutex, portMAX_DELAY) == pdTRUE)
            {
                // Assuming inference::img_copy is a structure or object containing the RGB data.
                JpegCompressRgb(
                    inference::img_copy->data(),
                    inference::img_width,
                    inference::img_height,
                    30, // Quality (adjust as needed)
                    &jpeg);
                xSemaphoreGive(shared::img_mutex);

                if (!jpeg.empty())
                {
                    // Send the JPEG data
                    size_t total_sent = 0;
                    size_t remaining = jpeg.size();
                    const uint8_t *data_ptr = jpeg.data();

                    while (remaining > 0)
                    {
                        int sent = sendto(udp_socket, data_ptr , std::min(static_cast<int>(1472), static_cast<int>(remaining)), remaining > 1472, reinterpret_cast<struct sockaddr *>(&dest_address), sizeof(dest_address));
                        if (sent == -1)
                        {
                            // Handle send error
                            break;
                        }
                        total_sent += sent;
                        remaining -= sent;
                        data_ptr += sent;
                    }
                }
            }
            vTaskDelay(pdMS_TO_TICKS(150)); // Delay 1 second
        }
    }

}
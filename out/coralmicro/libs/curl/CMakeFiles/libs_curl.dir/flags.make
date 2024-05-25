# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# compile ASM with /home/administrator/security-camera/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-gcc
# compile C with /home/administrator/security-camera/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-gcc
ASM_DEFINES = -DBUILDING_LIBCURL -DCMSIS_NN -DCORAL_MICRO_ARDUINO=0 -DCPU_MIMXRT1176CVM8A_cm7 -DDATA_SECTION_IS_CACHEABLE=1 -DDEBUG_CONSOLE_TRANSFER_NON_BLOCKING -DFSL_OSA_TASK_ENABLE -DFSL_SDK_ENABLE_DRIVER_CACHE_CONTROL=1 -DHAVE_CONFIG_H -DLFS_THREADSAFE -DLWIP_ENET_FLEXIBLE_CONFIGURATION -DLWIP_HTTPD_CGI_SSI -DLWIP_HTTPD_CUSTOM_FILES -DLWIP_HTTPD_DYNAMIC_FILE_READ -DLWIP_HTTPD_DYNAMIC_HEADERS -DLWIP_HTTPD_FILE_EXTENSION -DLWIP_HTTPD_SUPPORT_POST -DLWIP_POSIX_SOCKETS_IO_NAMES=0 -DMBEDTLS_CONFIG_FILE=\"third_party/modified/nxp/rt1176-sdk/ksdk_mbedtls_config.h\" -DMCMGR_HANDLE_EXCEPTIONS=1 -DSDK_DEBUGCONSOLE_UART -DSDK_DELAY_USE_DWT -DSDK_OS_FREE_RTOS -DSERIAL_PORT_TYPE_UART=1 -DSNTP_SERVER_DNS -DUSB_HOST_CONFIG_DFU -DUSE_RTOS=1 -DUSE_SDRAM -DWIFI_PSK=\"\" -DWIFI_SSID=\"MyAccessPoint\" -DXIP_BOOT_HEADER_ENABLE=1 -DXIP_EXTERNAL_FLASH=1 -D__STARTUP_CLEAR_BSS -D__STARTUP_INITIALIZE_NONCACHEDATA -D__USE_SHMEM

ASM_INCLUDES = /home/administrator/security-camera/coralmicro /home/administrator/security-camera/coralmicro/. /home/administrator/security-camera/coralmicro/libs/curl /home/administrator/security-camera/coralmicro/third_party/curl/include /home/administrator/security-camera/coralmicro/third_party/curl/lib /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/mbedtls/include /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/mbedtls/port/ksdk /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/mbedtls/library /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/cm7 /home/administrator/security-camera/coralmicro/libs/nxp/rt1176-sdk /home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/boards/evkmimxrt1170/xip /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/common_task /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/flash/nand /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/internal_flash /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/lists /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/log /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/mem_manager /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/messaging /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/osa /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/phy /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/serial_manager /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/uart /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176 /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/debug_console /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/str /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/xip /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/device /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host/class /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/include /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/phy /home/administrator/security-camera/coralmicro/libs/FreeRTOS/. /home/administrator/security-camera/coralmicro/third_party/modified/FreeRTOS /home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk/rtos/freertos/freertos_kernel/include /home/administrator/security-camera/coralmicro/third_party/freertos_kernel/include /home/administrator/security-camera/coralmicro/third_party/freertos_kernel/portable/GCC/ARM_CM4F /home/administrator/security-camera/coralmicro/third_party/CMSIS /home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/Core/Include /home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/DSP/Include /home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/NN/Include /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/port /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/src/include /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/src/include/compat/posix /home/administrator/security-camera/coralmicro/third_party/tflite-micro /home/administrator/security-camera/coralmicro/third_party/flatbuffers/include /home/administrator/security-camera/coralmicro/third_party/gemmlowp /home/administrator/security-camera/coralmicro/third_party/kissfft /home/administrator/security-camera/coralmicro/third_party/ruy /home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/littlefs

ASM_FLAGS = -O3 -DNDEBUG -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16

C_DEFINES = -DBUILDING_LIBCURL -DCMSIS_NN -DCORAL_MICRO_ARDUINO=0 -DCPU_MIMXRT1176CVM8A_cm7 -DDATA_SECTION_IS_CACHEABLE=1 -DDEBUG_CONSOLE_TRANSFER_NON_BLOCKING -DFSL_OSA_TASK_ENABLE -DFSL_SDK_ENABLE_DRIVER_CACHE_CONTROL=1 -DHAVE_CONFIG_H -DLFS_THREADSAFE -DLWIP_ENET_FLEXIBLE_CONFIGURATION -DLWIP_HTTPD_CGI_SSI -DLWIP_HTTPD_CUSTOM_FILES -DLWIP_HTTPD_DYNAMIC_FILE_READ -DLWIP_HTTPD_DYNAMIC_HEADERS -DLWIP_HTTPD_FILE_EXTENSION -DLWIP_HTTPD_SUPPORT_POST -DLWIP_POSIX_SOCKETS_IO_NAMES=0 -DMBEDTLS_CONFIG_FILE=\"third_party/modified/nxp/rt1176-sdk/ksdk_mbedtls_config.h\" -DMCMGR_HANDLE_EXCEPTIONS=1 -DSDK_DEBUGCONSOLE_UART -DSDK_DELAY_USE_DWT -DSDK_OS_FREE_RTOS -DSERIAL_PORT_TYPE_UART=1 -DSNTP_SERVER_DNS -DUSB_HOST_CONFIG_DFU -DUSE_RTOS=1 -DUSE_SDRAM -DWIFI_PSK=\"\" -DWIFI_SSID=\"MyAccessPoint\" -DXIP_BOOT_HEADER_ENABLE=1 -DXIP_EXTERNAL_FLASH=1 -D__STARTUP_CLEAR_BSS -D__STARTUP_INITIALIZE_NONCACHEDATA -D__USE_SHMEM

C_INCLUDES = -I/home/administrator/security-camera/coralmicro -I/home/administrator/security-camera/coralmicro/. -I/home/administrator/security-camera/coralmicro/libs/curl -I/home/administrator/security-camera/coralmicro/third_party/curl/include -I/home/administrator/security-camera/coralmicro/third_party/curl/lib -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/mbedtls/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/mbedtls/port/ksdk -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/mbedtls/library -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/cm7 -I/home/administrator/security-camera/coralmicro/libs/nxp/rt1176-sdk -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/boards/evkmimxrt1170/xip -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/common_task -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/flash/nand -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/internal_flash -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/lists -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/log -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/mem_manager -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/messaging -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/osa -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/phy -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/serial_manager -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/uart -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176 -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/debug_console -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/str -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/xip -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/device -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host/class -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/phy -I/home/administrator/security-camera/coralmicro/libs/FreeRTOS/. -I/home/administrator/security-camera/coralmicro/third_party/modified/FreeRTOS -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk/rtos/freertos/freertos_kernel/include -I/home/administrator/security-camera/coralmicro/third_party/freertos_kernel/include -I/home/administrator/security-camera/coralmicro/third_party/freertos_kernel/portable/GCC/ARM_CM4F -I/home/administrator/security-camera/coralmicro/third_party/CMSIS -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/Core/Include -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/DSP/Include -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/NN/Include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/port -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/src/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/src/include/compat/posix -I/home/administrator/security-camera/coralmicro/third_party/tflite-micro -I/home/administrator/security-camera/coralmicro/third_party/flatbuffers/include -I/home/administrator/security-camera/coralmicro/third_party/gemmlowp -I/home/administrator/security-camera/coralmicro/third_party/kissfft -I/home/administrator/security-camera/coralmicro/third_party/ruy -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/littlefs

C_FLAGS = -Wall -Wno-psabi -mthumb -fno-common -ffunction-sections -fdata-sections -ffreestanding -fno-builtin -mapcs-frame --specs=nano.specs --specs=nosys.specs -u _printf_float -ffile-prefix-map=/home/administrator/security-camera= -std=gnu99 -g -Os -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16


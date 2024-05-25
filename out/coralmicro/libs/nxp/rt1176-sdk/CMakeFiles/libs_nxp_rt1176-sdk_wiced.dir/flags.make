# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# compile ASM with /home/administrator/security-camera/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-gcc
# compile C with /home/administrator/security-camera/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-gcc
# compile CXX with /home/administrator/security-camera/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-g++
ASM_DEFINES = -DALTERNATE_***REMOVED*** -DCORAL_MICRO_ARDUINO=0 -DCPU_MIMXRT1176CVM8A_cm7 -DDATA_SECTION_IS_CACHEABLE=1 -DDEBUG_CONSOLE_TRANSFER_NON_BLOCKING -DEXTERNAL_DCT -DFSL_OSA_TASK_ENABLE -DFSL_SDK_ENABLE_DRIVER_CACHE_CONTROL=1 -DHIGH_SPEED_SDIO_CLOCK -DLWIP_ENET_FLEXIBLE_CONFIGURATION -DLWIP_POSIX_SOCKETS_IO_NAMES=0 -DNETWORK_LwIP -DNO_BOOTLOADER_REQUIRED -DNO_BUILD_BOOTLOADER -DSDK_DEBUGCONSOLE_UART -DSDK_DELAY_USE_DWT -DSDK_OS_FREE_RTOS -DSERIAL_PORT_TYPE_UART=1 -DSNTP_SERVER_DNS -DUSB_HOST_CONFIG_DFU -DUSES_RESOURCES_IN_EXTERNAL_STORAGE -DUSE_RTOS=1 -DUSE_SDRAM -DWICED -DWICED_PAYLOAD_MTU=8320 -DWICED_PLATFORM_MASKS_BUS_IRQ -DWIFI_PSK=\"\" -DWIFI_SSID=\"MyAccessPoint\" -DWLAN_ARM_CR4 -DWPRINT_ENABLE_WWD_ERROR -DWPRINT_PLATFORM_PERMISSION -DWWD_DOWNLOAD_CLM_BLOB -DXIP_BOOT_HEADER_ENABLE=1 -DXIP_EXTERNAL_FLASH=1 -D__STARTUP_CLEAR_BSS -D__STARTUP_INITIALIZE_NONCACHEDATA -D__USE_SHMEM

ASM_INCLUDES = -I/home/administrator/security-camera/coralmicro -I/home/administrator/security-camera/coralmicro/. -I/home/administrator/security-camera/coralmicro/libs/nxp/rt1176-sdk/WICED/platform -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk/rtos/freertos/libraries/abstractions/wifi/include -I/home/administrator/security-camera/coralmicro/libs/nxp/rt1176-sdk/rtos/freertos/libraries/abstractions/wifi/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/RTOS/FreeRTOS/WICED -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/RTOS/FreeRTOS/WWD -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/include/RTOS -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/include/network -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/internal/bus_protocols/SDIO -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/internal/chips/43455 -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/network/LwIP/WICED -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/network/LwIP/WWD -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/platform/ARM_CM7 -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/platform/GCC -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/platform/MCU -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/platform/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/P2P -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/WPS -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/crypto_internal -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/host/WICED -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/mbedtls_open/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/daemons/DHCP_server -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/protocols/DNS -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/utilities/TLV -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/utilities/linked_list -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/utilities/ring_buffer -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/utilities/wifi -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/platforms/coralmicro -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/rtos/freertos/libraries/c_sdk/standard/common/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/rtos/freertos/libraries/c_sdk/standard/common/include/types -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/cm7 -I/home/administrator/security-camera/coralmicro/libs/nxp/rt1176-sdk -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/boards/evkmimxrt1170/xip -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/common_task -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/flash/nand -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/internal_flash -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/lists -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/log -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/mem_manager -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/messaging -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/osa -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/phy -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/serial_manager -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/uart -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176 -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/debug_console -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/str -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/xip -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/device -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host/class -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/phy -I/home/administrator/security-camera/coralmicro/libs/FreeRTOS/. -I/home/administrator/security-camera/coralmicro/third_party/modified/FreeRTOS -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk/rtos/freertos/freertos_kernel/include -I/home/administrator/security-camera/coralmicro/third_party/freertos_kernel/include -I/home/administrator/security-camera/coralmicro/third_party/freertos_kernel/portable/GCC/ARM_CM4F -I/home/administrator/security-camera/coralmicro/third_party/CMSIS -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/Core/Include -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/DSP/Include -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/NN/Include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/port -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/src/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/src/include/compat/posix -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/common -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/host/usdhc -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/osa -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/sdio

ASM_FLAGS = -O3 -DNDEBUG -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16

C_DEFINES = -DALTERNATE_***REMOVED*** -DCORAL_MICRO_ARDUINO=0 -DCPU_MIMXRT1176CVM8A_cm7 -DDATA_SECTION_IS_CACHEABLE=1 -DDEBUG_CONSOLE_TRANSFER_NON_BLOCKING -DEXTERNAL_DCT -DFSL_OSA_TASK_ENABLE -DFSL_SDK_ENABLE_DRIVER_CACHE_CONTROL=1 -DHIGH_SPEED_SDIO_CLOCK -DLWIP_ENET_FLEXIBLE_CONFIGURATION -DLWIP_POSIX_SOCKETS_IO_NAMES=0 -DNETWORK_LwIP -DNO_BOOTLOADER_REQUIRED -DNO_BUILD_BOOTLOADER -DSDK_DEBUGCONSOLE_UART -DSDK_DELAY_USE_DWT -DSDK_OS_FREE_RTOS -DSERIAL_PORT_TYPE_UART=1 -DSNTP_SERVER_DNS -DUSB_HOST_CONFIG_DFU -DUSES_RESOURCES_IN_EXTERNAL_STORAGE -DUSE_RTOS=1 -DUSE_SDRAM -DWICED -DWICED_PAYLOAD_MTU=8320 -DWICED_PLATFORM_MASKS_BUS_IRQ -DWIFI_PSK=\"\" -DWIFI_SSID=\"MyAccessPoint\" -DWLAN_ARM_CR4 -DWPRINT_ENABLE_WWD_ERROR -DWPRINT_PLATFORM_PERMISSION -DWWD_DOWNLOAD_CLM_BLOB -DXIP_BOOT_HEADER_ENABLE=1 -DXIP_EXTERNAL_FLASH=1 -D__STARTUP_CLEAR_BSS -D__STARTUP_INITIALIZE_NONCACHEDATA -D__USE_SHMEM

C_INCLUDES = -I/home/administrator/security-camera/coralmicro -I/home/administrator/security-camera/coralmicro/. -I/home/administrator/security-camera/coralmicro/libs/nxp/rt1176-sdk/WICED/platform -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk/rtos/freertos/libraries/abstractions/wifi/include -I/home/administrator/security-camera/coralmicro/libs/nxp/rt1176-sdk/rtos/freertos/libraries/abstractions/wifi/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/RTOS/FreeRTOS/WICED -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/RTOS/FreeRTOS/WWD -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/include/RTOS -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/include/network -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/internal/bus_protocols/SDIO -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/internal/chips/43455 -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/network/LwIP/WICED -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/network/LwIP/WWD -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/platform/ARM_CM7 -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/platform/GCC -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/platform/MCU -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/platform/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/P2P -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/WPS -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/crypto_internal -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/host/WICED -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/mbedtls_open/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/daemons/DHCP_server -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/protocols/DNS -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/utilities/TLV -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/utilities/linked_list -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/utilities/ring_buffer -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/utilities/wifi -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/platforms/coralmicro -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/rtos/freertos/libraries/c_sdk/standard/common/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/rtos/freertos/libraries/c_sdk/standard/common/include/types -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/cm7 -I/home/administrator/security-camera/coralmicro/libs/nxp/rt1176-sdk -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/boards/evkmimxrt1170/xip -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/common_task -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/flash/nand -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/internal_flash -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/lists -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/log -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/mem_manager -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/messaging -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/osa -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/phy -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/serial_manager -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/uart -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176 -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/debug_console -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/str -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/xip -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/device -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host/class -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/phy -I/home/administrator/security-camera/coralmicro/libs/FreeRTOS/. -I/home/administrator/security-camera/coralmicro/third_party/modified/FreeRTOS -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk/rtos/freertos/freertos_kernel/include -I/home/administrator/security-camera/coralmicro/third_party/freertos_kernel/include -I/home/administrator/security-camera/coralmicro/third_party/freertos_kernel/portable/GCC/ARM_CM4F -I/home/administrator/security-camera/coralmicro/third_party/CMSIS -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/Core/Include -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/DSP/Include -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/NN/Include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/port -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/src/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/src/include/compat/posix -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/common -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/host/usdhc -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/osa -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/sdio

C_FLAGS = -Wall -Wno-psabi -mthumb -fno-common -ffunction-sections -fdata-sections -ffreestanding -fno-builtin -mapcs-frame --specs=nano.specs --specs=nosys.specs -u _printf_float -ffile-prefix-map=/home/administrator/security-camera= -std=gnu99 -g -Os -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16

CXX_DEFINES = -DALTERNATE_***REMOVED*** -DCORAL_MICRO_ARDUINO=0 -DCPU_MIMXRT1176CVM8A_cm7 -DDATA_SECTION_IS_CACHEABLE=1 -DDEBUG_CONSOLE_TRANSFER_NON_BLOCKING -DEXTERNAL_DCT -DFSL_OSA_TASK_ENABLE -DFSL_SDK_ENABLE_DRIVER_CACHE_CONTROL=1 -DHIGH_SPEED_SDIO_CLOCK -DLWIP_ENET_FLEXIBLE_CONFIGURATION -DLWIP_POSIX_SOCKETS_IO_NAMES=0 -DNETWORK_LwIP -DNO_BOOTLOADER_REQUIRED -DNO_BUILD_BOOTLOADER -DSDK_DEBUGCONSOLE_UART -DSDK_DELAY_USE_DWT -DSDK_OS_FREE_RTOS -DSERIAL_PORT_TYPE_UART=1 -DSNTP_SERVER_DNS -DUSB_HOST_CONFIG_DFU -DUSES_RESOURCES_IN_EXTERNAL_STORAGE -DUSE_RTOS=1 -DUSE_SDRAM -DWICED -DWICED_PAYLOAD_MTU=8320 -DWICED_PLATFORM_MASKS_BUS_IRQ -DWIFI_PSK=\"\" -DWIFI_SSID=\"MyAccessPoint\" -DWLAN_ARM_CR4 -DWPRINT_ENABLE_WWD_ERROR -DWPRINT_PLATFORM_PERMISSION -DWWD_DOWNLOAD_CLM_BLOB -DXIP_BOOT_HEADER_ENABLE=1 -DXIP_EXTERNAL_FLASH=1 -D__STARTUP_CLEAR_BSS -D__STARTUP_INITIALIZE_NONCACHEDATA -D__USE_SHMEM

CXX_INCLUDES = -I/home/administrator/security-camera/coralmicro -I/home/administrator/security-camera/coralmicro/. -I/home/administrator/security-camera/coralmicro/libs/nxp/rt1176-sdk/WICED/platform -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk/rtos/freertos/libraries/abstractions/wifi/include -I/home/administrator/security-camera/coralmicro/libs/nxp/rt1176-sdk/rtos/freertos/libraries/abstractions/wifi/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/RTOS/FreeRTOS/WICED -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/RTOS/FreeRTOS/WWD -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/include/RTOS -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/include/network -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/internal/bus_protocols/SDIO -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/WWD/internal/chips/43455 -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/network/LwIP/WICED -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/network/LwIP/WWD -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/platform/ARM_CM7 -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/platform/GCC -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/platform/MCU -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/platform/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/P2P -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/WPS -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/crypto_internal -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/host/WICED -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/WICED/security/BESL/mbedtls_open/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/daemons/DHCP_server -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/protocols/DNS -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/utilities/TLV -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/utilities/linked_list -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/utilities/ring_buffer -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/libraries/utilities/wifi -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/wiced/43xxx_Wi-Fi/platforms/coralmicro -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/rtos/freertos/libraries/c_sdk/standard/common/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/rtos/freertos/libraries/c_sdk/standard/common/include/types -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/cm7 -I/home/administrator/security-camera/coralmicro/libs/nxp/rt1176-sdk -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/boards/evkmimxrt1170/xip -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/common_task -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/flash/nand -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/internal_flash -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/lists -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/log -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/mem_manager -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/messaging -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/osa -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/phy -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/serial_manager -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/uart -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176 -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/debug_console -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/str -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/xip -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/device -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host/class -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/phy -I/home/administrator/security-camera/coralmicro/libs/FreeRTOS/. -I/home/administrator/security-camera/coralmicro/third_party/modified/FreeRTOS -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk/rtos/freertos/freertos_kernel/include -I/home/administrator/security-camera/coralmicro/third_party/freertos_kernel/include -I/home/administrator/security-camera/coralmicro/third_party/freertos_kernel/portable/GCC/ARM_CM4F -I/home/administrator/security-camera/coralmicro/third_party/CMSIS -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/Core/Include -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/DSP/Include -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/NN/Include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/port -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/src/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/lwip/src/include/compat/posix -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/common -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/host/usdhc -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/osa -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/sdio

CXX_FLAGS = -Wall -Wno-psabi -mthumb -fno-common -ffunction-sections -fdata-sections -ffreestanding -fno-builtin -mapcs-frame --specs=nano.specs --specs=nosys.specs -u _printf_float -ffile-prefix-map=/home/administrator/security-camera= -fno-rtti -fno-exceptions -g -Os -std=gnu++17 -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16


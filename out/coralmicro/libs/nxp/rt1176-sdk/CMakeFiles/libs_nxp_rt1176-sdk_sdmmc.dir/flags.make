# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# compile ASM with /home/administrator/security-camera/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-gcc
# compile C with /home/administrator/security-camera/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-gcc
ASM_DEFINES = -DCORAL_MICRO_ARDUINO=0 -DCPU_MIMXRT1176CVM8A_cm7 -DDATA_SECTION_IS_CACHEABLE=1 -DDEBUG_CONSOLE_TRANSFER_NON_BLOCKING -DFSL_OSA_TASK_ENABLE -DFSL_SDK_ENABLE_DRIVER_CACHE_CONTROL=1 -DLWIP_ENET_FLEXIBLE_CONFIGURATION -DLWIP_POSIX_SOCKETS_IO_NAMES=0 -DSDK_DEBUGCONSOLE_UART -DSDK_DELAY_USE_DWT -DSDK_OS_FREE_RTOS -DSERIAL_PORT_TYPE_UART=1 -DUSB_HOST_CONFIG_DFU -DUSE_RTOS=1 -DUSE_SDRAM -DWIFI_PSK=\"\" -DWIFI_SSID=\"MyAccessPoint\" -DXIP_BOOT_HEADER_ENABLE=1 -DXIP_EXTERNAL_FLASH=1 -D__STARTUP_CLEAR_BSS -D__STARTUP_INITIALIZE_NONCACHEDATA -D__USE_SHMEM

ASM_INCLUDES = -I/home/administrator/security-camera/coralmicro -I/home/administrator/security-camera/coralmicro/. -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/common -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/host/usdhc -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/osa -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/sdio -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/cm7 -I/home/administrator/security-camera/coralmicro/libs/nxp/rt1176-sdk -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/boards/evkmimxrt1170/xip -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/common_task -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/flash/nand -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/internal_flash -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/lists -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/log -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/mem_manager -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/messaging -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/osa -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/phy -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/serial_manager -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/uart -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176 -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/debug_console -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/str -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/xip -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/device -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host/class -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/phy -I/home/administrator/security-camera/coralmicro/libs/FreeRTOS/. -I/home/administrator/security-camera/coralmicro/third_party/modified/FreeRTOS -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk/rtos/freertos/freertos_kernel/include -I/home/administrator/security-camera/coralmicro/third_party/freertos_kernel/include -I/home/administrator/security-camera/coralmicro/third_party/freertos_kernel/portable/GCC/ARM_CM4F -I/home/administrator/security-camera/coralmicro/third_party/CMSIS -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/Core/Include -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/DSP/Include -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/NN/Include

ASM_FLAGS = -O3 -DNDEBUG -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16

C_DEFINES = -DCORAL_MICRO_ARDUINO=0 -DCPU_MIMXRT1176CVM8A_cm7 -DDATA_SECTION_IS_CACHEABLE=1 -DDEBUG_CONSOLE_TRANSFER_NON_BLOCKING -DFSL_OSA_TASK_ENABLE -DFSL_SDK_ENABLE_DRIVER_CACHE_CONTROL=1 -DLWIP_ENET_FLEXIBLE_CONFIGURATION -DLWIP_POSIX_SOCKETS_IO_NAMES=0 -DSDK_DEBUGCONSOLE_UART -DSDK_DELAY_USE_DWT -DSDK_OS_FREE_RTOS -DSERIAL_PORT_TYPE_UART=1 -DUSB_HOST_CONFIG_DFU -DUSE_RTOS=1 -DUSE_SDRAM -DWIFI_PSK=\"\" -DWIFI_SSID=\"MyAccessPoint\" -DXIP_BOOT_HEADER_ENABLE=1 -DXIP_EXTERNAL_FLASH=1 -D__STARTUP_CLEAR_BSS -D__STARTUP_INITIALIZE_NONCACHEDATA -D__USE_SHMEM

C_INCLUDES = -I/home/administrator/security-camera/coralmicro -I/home/administrator/security-camera/coralmicro/. -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/common -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/host/usdhc -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/osa -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/sdmmc/sdio -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/cm7 -I/home/administrator/security-camera/coralmicro/libs/nxp/rt1176-sdk -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/boards/evkmimxrt1170/xip -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/common_task -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/flash/nand -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/internal_flash -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/lists -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/log -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/mem_manager -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/messaging -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/osa -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/phy -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/serial_manager -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/components/uart -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176 -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/debug_console -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/str -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/xip -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/device -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/host/class -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/include -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class -I/home/administrator/security-camera/coralmicro/third_party/nxp/rt1176-sdk/middleware/usb/phy -I/home/administrator/security-camera/coralmicro/libs/FreeRTOS/. -I/home/administrator/security-camera/coralmicro/third_party/modified/FreeRTOS -I/home/administrator/security-camera/coralmicro/third_party/modified/nxp/rt1176-sdk/rtos/freertos/freertos_kernel/include -I/home/administrator/security-camera/coralmicro/third_party/freertos_kernel/include -I/home/administrator/security-camera/coralmicro/third_party/freertos_kernel/portable/GCC/ARM_CM4F -I/home/administrator/security-camera/coralmicro/third_party/CMSIS -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/Core/Include -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/DSP/Include -I/home/administrator/security-camera/coralmicro/third_party/CMSIS/CMSIS/NN/Include

C_FLAGS = -Wall -Wno-psabi -mthumb -fno-common -ffunction-sections -fdata-sections -ffreestanding -fno-builtin -mapcs-frame --specs=nano.specs --specs=nosys.specs -u _printf_float -ffile-prefix-map=/home/administrator/security-camera= -std=gnu99 -g -Os -mcpu=cortex-m7 -mfloat-abi=hard -mfpu=fpv5-d16


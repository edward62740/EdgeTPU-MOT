# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.29

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/administrator/.local/lib/python3.8/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/administrator/.local/lib/python3.8/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/administrator/security-camera

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/administrator/security-camera/out

# Include any dependencies generated for this target.
include coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/compiler_depend.make

# Include the progress variables for this target.
include coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/progress.make

# Include the compile flags for this target's objects.
include coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/flags.make

coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.obj: coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/flags.make
coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.obj: /home/administrator/security-camera/coralmicro/libs/libjpeg/jpeg.cc
coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.obj: coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/administrator/security-camera/out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.obj"
	cd /home/administrator/security-camera/out/coralmicro/libs/libjpeg && /home/administrator/security-camera/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.obj -MF CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.obj.d -o CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.obj -c /home/administrator/security-camera/coralmicro/libs/libjpeg/jpeg.cc

coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.i"
	cd /home/administrator/security-camera/out/coralmicro/libs/libjpeg && /home/administrator/security-camera/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/administrator/security-camera/coralmicro/libs/libjpeg/jpeg.cc > CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.i

coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.s"
	cd /home/administrator/security-camera/out/coralmicro/libs/libjpeg && /home/administrator/security-camera/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/administrator/security-camera/coralmicro/libs/libjpeg/jpeg.cc -o CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.s

# Object files for target libs_jpeg_m7
libs_jpeg_m7_OBJECTS = \
"CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.obj"

# External object files for target libs_jpeg_m7
libs_jpeg_m7_EXTERNAL_OBJECTS = \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/osa/fsl_os_abstraction_free_rtos.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpi2c_freertos.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpspi_freertos.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpuart_freertos.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/modified/nxp/rt1176-sdk/fsl_tickless_gpt.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/cm7/fsl_cache.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/gcc/startup_MIMXRT1176_cm7.S.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/system_MIMXRT1176_cm7.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/board_hardware.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/clock_config.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/usb_device_cdc_eem.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class/usb_device_msc.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class/usb_device_msc_ufi.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/modified/nxp/rt1176-sdk/board.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/modified/nxp/rt1176-sdk/pin_mux.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/modified/nxp/rt1176-sdk/usb_device_class.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/flash/nand/flexspi/fsl_flexspi_nand_flash.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/lists/fsl_component_generic_list.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/phy/device/phyrtl8211f/fsl_phyrtl8211f.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/phy/mdio/enet/fsl_enet_mdio.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/serial_manager/fsl_component_serial_manager.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/serial_manager/fsl_component_serial_port_uart.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/uart/fsl_adapter_lpuart.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_anatop_ai.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_caam.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_clock.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_common.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_common_arm.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_csi.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_dac12.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_dcdc.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_dmamux.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_edma.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_enet.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_flexspi.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_gpio.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_gpc.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_gpt.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpadc.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpi2c.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpspi.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpuart.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_mu.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_ocotp.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pdm.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pdm_edma.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pmu.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pwm.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pxp.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_romapi.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_sema4.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_semc.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_snvs_hp.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_snvs_lp.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_soc_src.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_tempsensor.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_wdog.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_xbara.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/debug_console/fsl_debug_console.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/str/fsl_str.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/device/usb_device_dci.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/device/usb_device_ehci.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/class/usb_host_dfu.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/class/usb_host_hub.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/class/usb_host_hub_app.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/usb_host_devices.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/usb_host_ehci.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/usb_host_framework.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/usb_host_hci.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class/usb_device_cdc_acm.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class/usb_device_hid.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/usb_device_ch9.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/phy/usb_phy.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/app_callbacks.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/multicore.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/event_groups.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/list.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/portable/GCC/ARM_CM4F/port.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/queue.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/stream_buffer.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/tasks.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/timers.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/FreeRTOS_helpers/heap_useNewlib_NXP.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk-mcmgr_m7.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/multicore/mcmgr/src/mcmgr.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk-mcmgr_m7.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/multicore/mcmgr/src/mcmgr_internal_core_api_imxrt1170.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk-mcmgr_m7.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/multicore/mcmgr/src/mcmgr_mu_internal.c.obj"

coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/jpeg.cc.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/osa/fsl_os_abstraction_free_rtos.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpi2c_freertos.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpspi_freertos.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpuart_freertos.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/modified/nxp/rt1176-sdk/fsl_tickless_gpt.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/cm7/fsl_cache.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/gcc/startup_MIMXRT1176_cm7.S.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/system_MIMXRT1176_cm7.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/board_hardware.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/clock_config.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/usb_device_cdc_eem.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class/usb_device_msc.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class/usb_device_msc_ufi.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/modified/nxp/rt1176-sdk/board.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/modified/nxp/rt1176-sdk/pin_mux.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/modified/nxp/rt1176-sdk/usb_device_class.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/flash/nand/flexspi/fsl_flexspi_nand_flash.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/lists/fsl_component_generic_list.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/phy/device/phyrtl8211f/fsl_phyrtl8211f.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/phy/mdio/enet/fsl_enet_mdio.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/serial_manager/fsl_component_serial_manager.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/serial_manager/fsl_component_serial_port_uart.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/components/uart/fsl_adapter_lpuart.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_anatop_ai.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_caam.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_clock.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_common.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_common_arm.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_csi.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_dac12.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_dcdc.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_dmamux.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_edma.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_enet.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_flexspi.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_gpio.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_gpc.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_gpt.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpadc.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpi2c.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpspi.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_lpuart.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_mu.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_ocotp.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pdm.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pdm_edma.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pmu.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pwm.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_pxp.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_romapi.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_sema4.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_semc.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_snvs_hp.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_snvs_lp.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_soc_src.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_tempsensor.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_wdog.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/drivers/fsl_xbara.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/debug_console/fsl_debug_console.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/devices/MIMXRT1176/utilities/str/fsl_str.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/device/usb_device_dci.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/device/usb_device_ehci.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/class/usb_host_dfu.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/class/usb_host_hub.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/class/usb_host_hub_app.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/usb_host_devices.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/usb_host_ehci.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/usb_host_framework.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/host/usb_host_hci.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class/usb_device_cdc_acm.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/class/usb_device_hid.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/output/source/device/usb_device_ch9.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk_freertos.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/usb/phy/usb_phy.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/app_callbacks.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/multicore.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/event_groups.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/list.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/portable/GCC/ARM_CM4F/port.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/queue.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/stream_buffer.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/tasks.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/timers.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/FreeRTOS_helpers/heap_useNewlib_NXP.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk-mcmgr_m7.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/multicore/mcmgr/src/mcmgr.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk-mcmgr_m7.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/multicore/mcmgr/src/mcmgr_internal_core_api_imxrt1170.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/nxp/rt1176-sdk/CMakeFiles/libs_nxp_rt1176-sdk-mcmgr_m7.dir/__/__/__/third_party/nxp/rt1176-sdk/middleware/multicore/mcmgr/src/mcmgr_mu_internal.c.obj
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/build.make
coralmicro/libs/libjpeg/liblibs_jpeg_m7.a: coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/administrator/security-camera/out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library liblibs_jpeg_m7.a"
	cd /home/administrator/security-camera/out/coralmicro/libs/libjpeg && $(CMAKE_COMMAND) -P CMakeFiles/libs_jpeg_m7.dir/cmake_clean_target.cmake
	cd /home/administrator/security-camera/out/coralmicro/libs/libjpeg && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libs_jpeg_m7.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/build: coralmicro/libs/libjpeg/liblibs_jpeg_m7.a
.PHONY : coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/build

coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/clean:
	cd /home/administrator/security-camera/out/coralmicro/libs/libjpeg && $(CMAKE_COMMAND) -P CMakeFiles/libs_jpeg_m7.dir/cmake_clean.cmake
.PHONY : coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/clean

coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/depend:
	cd /home/administrator/security-camera/out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/administrator/security-camera /home/administrator/security-camera/coralmicro/libs/libjpeg /home/administrator/security-camera/out /home/administrator/security-camera/out/coralmicro/libs/libjpeg /home/administrator/security-camera/out/coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : coralmicro/libs/libjpeg/CMakeFiles/libs_jpeg_m7.dir/depend


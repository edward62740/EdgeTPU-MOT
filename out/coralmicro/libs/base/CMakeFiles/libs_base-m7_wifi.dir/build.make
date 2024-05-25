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
include coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/compiler_depend.make

# Include the progress variables for this target.
include coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/progress.make

# Include the compile flags for this target's objects.
include coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/flags.make

coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.obj: coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/flags.make
coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.obj: /home/administrator/security-camera/coralmicro/libs/base/wifi.cc
coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.obj: coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/administrator/security-camera/out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.obj"
	cd /home/administrator/security-camera/out/coralmicro/libs/base && /home/administrator/security-camera/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.obj -MF CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.obj.d -o CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.obj -c /home/administrator/security-camera/coralmicro/libs/base/wifi.cc

coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.i"
	cd /home/administrator/security-camera/out/coralmicro/libs/base && /home/administrator/security-camera/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/administrator/security-camera/coralmicro/libs/base/wifi.cc > CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.i

coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.s"
	cd /home/administrator/security-camera/out/coralmicro/libs/base && /home/administrator/security-camera/coralmicro/third_party/toolchain-linux/gcc-arm-none-eabi-9-2020-q2-update/bin/arm-none-eabi-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/administrator/security-camera/coralmicro/libs/base/wifi.cc -o CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.s

# Object files for target libs_base-m7_wifi
libs_base__m7_wifi_OBJECTS = \
"CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.obj"

# External object files for target libs_base-m7_wifi
libs_base__m7_wifi_EXTERNAL_OBJECTS = \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/app_callbacks.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/multicore.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/event_groups.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/list.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/portable/GCC/ARM_CM4F/port.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/queue.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/stream_buffer.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/tasks.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/timers.c.obj" \
"/home/administrator/security-camera/out/coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/FreeRTOS_helpers/heap_useNewlib_NXP.c.obj"

coralmicro/libs/base/liblibs_base-m7_wifi.a: coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/wifi.cc.obj
coralmicro/libs/base/liblibs_base-m7_wifi.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/app_callbacks.c.obj
coralmicro/libs/base/liblibs_base-m7_wifi.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/multicore.c.obj
coralmicro/libs/base/liblibs_base-m7_wifi.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/event_groups.c.obj
coralmicro/libs/base/liblibs_base-m7_wifi.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/list.c.obj
coralmicro/libs/base/liblibs_base-m7_wifi.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/portable/GCC/ARM_CM4F/port.c.obj
coralmicro/libs/base/liblibs_base-m7_wifi.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/queue.c.obj
coralmicro/libs/base/liblibs_base-m7_wifi.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/stream_buffer.c.obj
coralmicro/libs/base/liblibs_base-m7_wifi.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/tasks.c.obj
coralmicro/libs/base/liblibs_base-m7_wifi.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/freertos_kernel/timers.c.obj
coralmicro/libs/base/liblibs_base-m7_wifi.a: coralmicro/libs/FreeRTOS/CMakeFiles/libs_FreeRTOS.dir/__/__/third_party/FreeRTOS_helpers/heap_useNewlib_NXP.c.obj
coralmicro/libs/base/liblibs_base-m7_wifi.a: coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/build.make
coralmicro/libs/base/liblibs_base-m7_wifi.a: coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/administrator/security-camera/out/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library liblibs_base-m7_wifi.a"
	cd /home/administrator/security-camera/out/coralmicro/libs/base && $(CMAKE_COMMAND) -P CMakeFiles/libs_base-m7_wifi.dir/cmake_clean_target.cmake
	cd /home/administrator/security-camera/out/coralmicro/libs/base && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/libs_base-m7_wifi.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/build: coralmicro/libs/base/liblibs_base-m7_wifi.a
.PHONY : coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/build

coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/clean:
	cd /home/administrator/security-camera/out/coralmicro/libs/base && $(CMAKE_COMMAND) -P CMakeFiles/libs_base-m7_wifi.dir/cmake_clean.cmake
.PHONY : coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/clean

coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/depend:
	cd /home/administrator/security-camera/out && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/administrator/security-camera /home/administrator/security-camera/coralmicro/libs/base /home/administrator/security-camera/out /home/administrator/security-camera/out/coralmicro/libs/base /home/administrator/security-camera/out/coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : coralmicro/libs/base/CMakeFiles/libs_base-m7_wifi.dir/depend


#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(Cheetah-Vision)

# Save the command line compile commands in the build output
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

if(BUILD_GRAPHICAL_EXAMPLES)
    include(${CMAKE_SOURCE_DIR}/CMake/opengl_config.cmake)
else()
    if(ANDROID_NDK_TOOLCHAIN_INCLUDED)
        find_library(log-lib log)
        set(DEPENDENCIES realsense2 log)
    else()
        set(DEPENDENCIES realsense2)
    endif()
endif()

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

#pointcloud
add_executable(cheetah-vision cheetah_pointcloud.cpp)

set_property(TARGET cheetah-vision PROPERTY CXX_STANDARD 11)

include_directories(../../Cheetah-Software/lcm-types/cpp)
target_link_libraries(cheetah-vision ${DEPENDENCIES})
target_link_libraries(cheetah-vision lcm)

find_package( OpenCV REQUIRED )
message(${OpenCV_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(cheetah-vision ${OpenCV_LIBS} realsense2-gl)


cmake_minimum_required(VERSION 3.22)
project(ffmpeg_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-D__STDC_CONSTANT_MACROS")

set(CMAKE_BUILD_TYPE Debug)
include_directories(./include)
include_directories(/usr/local/cuda-10.2/include/)
link_directories(/usr/local/cuda-10.2/lib64/)

##########
##ffmpeg##
##########
find_path(AVCODEC_INCLUDE_DIR libavcodec/avcodec.h)
find_library(AVCODEC_LIBRARY avcodec)

find_path(AVFORMAT_INCLUDE_DIR libavformat/avformat.h)
find_library(AVFORMAT_LIBRARY avformat)

find_path(AVUTIL_INCLUDE_DIR libavutil/avutil.h)
find_library(AVUTIL_LIBRARY avutil)

find_path(AVDEVICE_INCLUDE_DIR libavdevice/avdevice.h)
find_library(AVDEVICE_LIBRARY avdevice)
##########
##########


##########
##opencv##
##########
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
##########
##########


##########
##glog####
##########
find_package(glog 0.6.0 REQUIRED)
##########
##########

add_executable(ffmpeg_test source/hw_decode.cpp test.cpp)
target_include_directories(ffmpeg_test PRIVATE ${AVCODEC_INCLUDE_DIR} ${AVFORMAT_INCLUDE_DIR} ${AVUTIL_INCLUDE_DIR} ${AVDEVICE_INCLUDE_DIR})
target_link_libraries(ffmpeg_test PRIVATE ${AVSWS_LIBRARY} ${AVCODEC_LIBRARY} ${AVFORMAT_LIBRARY} ${AVUTIL_LIBRARY} ${AVDEVICE_LIBRARY} ${OpenCV_LIBS} cudart glog::glog)


cmake_minimum_required(VERSION 3.28)
project(test04 VERSION 1.0 LANGUAGES CXX)

find_package(PkgConfig REQUIRED)
find_package(OpenCV REQUIRED)

# Find GStreamer components and create imported targets
pkg_search_module(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.24)
pkg_search_module(GSTAPP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
pkg_search_module(GSTVIDEO REQUIRED IMPORTED_TARGET gstreamer-video-1.0)

set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_LIST_DIR}/../test02/3rdparty/onnxruntime-linux-x64-gpu-1.26.0")

add_executable(${PROJECT_NAME} main.cpp)

include_directories(${PROJECT_NAME} ${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

target_link_libraries(${PROJECT_NAME} PRIVATE 
    PkgConfig::GSTREAMER 
    PkgConfig::GSTAPP 
    PkgConfig::GSTVIDEO 
    ${OpenCV_LIBS} 
    ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so
)

cmake_minimum_required (VERSION 3.28)

project ("test02" VERSION 1.0 LANGUAGES CXX)
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_LIST_DIR}/3rdparty/onnxruntime-linux-x64-gpu-1.26.0")

add_executable (${PROJECT_NAME} "main.cpp")

find_package(OpenCV REQUIRED)
include_directories(${PROJECT_NAME} ${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)



cmake_minimum_required (VERSION 3.22)

project ("test05" VERSION 1.0 LANGUAGES CXX)
set(ONNXRUNTIME_DIR "/home/huynq/libs/onnxruntime-linux-x64-1.26.0")

add_executable (${PROJECT_NAME} "main.cpp")

find_package(OpenCV REQUIRED)
include_directories(${PROJECT_NAME} ${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${ONNXRUNTIME_DIR}/lib/libonnxruntime.so)



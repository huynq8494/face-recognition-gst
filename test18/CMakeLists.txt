cmake_minimum_required(VERSION 3.28)
project(test18 VERSION 1.0 LANGUAGES CXX)

if (ENABLE_ASAN)
	message(STATUS "BUILD WITH ADDRESS SANITIZER")
	set (CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
	set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
endif ()
set(rknpu_scrfd_file rknpu2/scrfd_rknn.cc)
add_subdirectory(/home/hw/rknn_model_zoo/3rdparty/ 3rdparty.out)
add_subdirectory(/home/hw/rknn_model_zoo/utils/ utils.out)

find_package(PkgConfig REQUIRED)
pkg_search_module(GSTREAMER REQUIRED IMPORTED_TARGET gstreamer-1.0>=1.20)
pkg_search_module(GSTAPP REQUIRED IMPORTED_TARGET gstreamer-app-1.0)
pkg_search_module(GSTVIDEO REQUIRED IMPORTED_TARGET gstreamer-video-1.0)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(${PROJECT_NAME} 
    main.cpp
    ${rknpu_scrfd_file})
target_link_libraries(${PROJECT_NAME}
    PkgConfig::GSTREAMER
    PkgConfig::GSTAPP
    PkgConfig::GSTVIDEO
    rga
    imageutils
    fileutils
    imagedrawing    
    ${LIBRKNNRT}
    dl
    Threads::Threads)
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${LIBRKNNRT_INCLUDES}
)

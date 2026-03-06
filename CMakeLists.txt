cmake_minimum_required(VERSION 3.10)
project(yolo_onnx_cpp_camera_gpu)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found!")
else()
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV version: ${OpenCV_VERSION}")
endif()

# CUDA
find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit not found!")
else()
    message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")
endif()


# 优先使用环境变量 TENSORRT_DIR，否则使用默认系统路径
if(DEFINED ENV{TENSORRT_DIR})
    set(TENSORRT_DIR $ENV{TENSORRT_DIR})
    set(TENSORRT_INCLUDE_DIR "${TENSORRT_DIR}/include")
    set(TENSORRT_LIB_DIR "${TENSORRT_DIR}/lib")
    message(STATUS "Using TensorRT from environment: ${TENSORRT_DIR}")
else()
    # 默认系统路径（apt 安装）
    set(TENSORRT_INCLUDE_DIR "/usr/include/x86_64-linux-gnu/")
    set(TENSORRT_LIB_DIR "/usr/lib/x86_64-linux-gnu/")
    message(STATUS "Using system default TensorRT path")
endif()

# 验证 TensorRT 库是否存在
find_path(TENSORRT_INCLUDE_PATH NvInfer.h PATHS ${TENSORRT_INCLUDE_DIR})
find_library(TENSORRT_NVINFER_LIB nvinfer PATHS ${TENSORRT_LIB_DIR})
find_library(TENSORRT_NVONNXPARSER_LIB nvonnxparser PATHS ${TENSORRT_LIB_DIR})

if(NOT TENSORRT_INCLUDE_PATH)
    message(FATAL_ERROR "TensorRT headers not found! Check TENSORRT_INCLUDE_DIR")
endif()
if(NOT TENSORRT_NVINFER_LIB OR NOT TENSORRT_NVONNXPARSER_LIB)
    message(FATAL_ERROR "TensorRT libraries not found! Check TENSORRT_LIB_DIR")
endif()
message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_PATH}")
message(STATUS "TensorRT Lib: ${TENSORRT_LIB_DIR}")


# 源文件
set(SOURCES
    src/preprocessor.cpp
    src/inference.cpp
    src/postprocessor.cpp
    src/visualizer.cpp
    src/detector.cpp
    main.cpp
)

# 创建可执行文件
add_executable(yolo_infer ${SOURCES})

target_include_directories(yolo_infer PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_PATH}
    ${PROJECT_SOURCE_DIR}/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(yolo_infer PRIVATE
    ${OpenCV_LIBS} 
    ${TENSORRT_NVINFER_LIB}
    ${TENSORRT_NVONNXPARSER_LIB}
    CUDA::cudart
)

set_target_properties(yolo_infer PROPERTIES
    BUILD_RPATH "${TENSORRT_LIB_DIR}"
    INSTALL_RPATH "${TENSORRT_LIB_DIR}"
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

# add_compile_options(-Wno-deprecated-declarations)
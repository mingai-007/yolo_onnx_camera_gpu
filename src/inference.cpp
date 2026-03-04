#include "inference.h"
#include <iostream>

Inference::Inference(const std::string& modelPath) 
    : env_(ORT_LOGGING_LEVEL_WARNING, "YOLO"),
      allocator_() {  // 初始化 allocator
    
    // 配置 Session 选项
    Ort::SessionOptions sessionOptions;     
    sessionOptions.SetIntraOpNumThreads(1);     // 单算子内并行线程数
    sessionOptions.SetInterOpNumThreads(1);     // 算子间并行线程数
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);   // 启用所有图优化
    
    OrtCUDAProviderOptions cuda_options;                        
    sessionOptions.AppendExecutionProvider_CUDA(cuda_options);    // 使用 CUDA 作为执行提供程序
    
    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions); 
    std::cout << "Model loaded successfully!" << std::endl;

    // 获取输入名称并使用智能指针管理其生命周期
    auto inputNameAllocated = session_->GetInputNameAllocated(0, allocator_);   
    size_t inputNameLen = strlen(inputNameAllocated.get()) + 1;
    inputNamePtr_ = std::make_unique<char[]>(inputNameLen);
    memcpy(inputNamePtr_.get(), inputNameAllocated.get(), inputNameLen);
    inputNameStr_ = inputNamePtr_.get();  // 保存 string 供使用
    
    // 获取输出名称
    auto outputNameAllocated = session_->GetOutputNameAllocated(0, allocator_);
    size_t outputNameLen = strlen(outputNameAllocated.get()) + 1;
    outputNamePtr_ = std::make_unique<char[]>(outputNameLen);
    memcpy(outputNamePtr_.get(), outputNameAllocated.get(), outputNameLen);
    outputNameStr_ = outputNamePtr_.get();
    
    // 获取输出形状
    auto outputTypeInfo = session_->GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    outputShape_ = outputTensorInfo.GetShape();
    
    // std::cout << "Input name: " << inputNameStr_ << std::endl;
    // std::cout << "Output name: " << outputNameStr_ << std::endl;
    std::cout << "Output shape: [";
    for (size_t i = 0; i < outputShape_.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << outputShape_[i];
    }
    std::cout << "]" << std::endl;
}

std::vector<float> Inference::run(const cv::Mat& inputBlob) {

    // std::cout << "Running inference with input blob of shape: [";
    // for (int d = 0; d < inputBlob.dims; ++d) {
    //     std::cout << inputBlob.size[d];
    //     if (d < inputBlob.dims - 1) std::cout << ", ";
    // }
    // std::cout << "]" << std::endl;

    // 确保输入是连续的内存
    cv::Mat continuousBlob;
    if (!inputBlob.isContinuous()) {
        continuousBlob = inputBlob.clone();
    } else {
        continuousBlob = inputBlob;
    }
    
    // 输入维度 [1, 3, H, W]
    std::vector<int64_t> inputDims(4);
    for (int d = 0; d < 4; ++d) {
        inputDims[d] = static_cast<int64_t>(continuousBlob.size[d]);
    }
    
    // std::cout << "Input dimensions for ONNX Runtime: ["
    //           << inputDims[0] << ", " << inputDims[1] << ", "
    //           << inputDims[2] << ", " << inputDims[3] << "]" <<  std::endl;
    
    // 计算元素总数
    size_t inputElementCount = continuousBlob.total();
    
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);  // 创建 CPU 内存信息对象
    // 改用 CUDA 内存
    // Ort::MemoryInfo memoryInfo("Cuda", OrtArenaAllocator, 0, OrtMemTypeDefault);
    
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,                                    // 内存信息
        reinterpret_cast<float*>(continuousBlob.data), // 输入数据指针
        inputElementCount,                             // 元素总数
        inputDims.data(),                              // 输入维度数组指针
        inputDims.size()                               // 输入维度数量
    );
    
    // 准备输入输出名称（必须使用持久化的字符串指针）
    const char* inputNames[] = {inputNameStr_.c_str()};
    const char* outputNames[] = {outputNameStr_.c_str()};
    
    // 执行推理
    auto outputTensors = session_->Run(
        Ort::RunOptions{nullptr},       
        inputNames, &inputTensor, 1,    // 输入：名称数组 + tensor 数组 + 数量
        outputNames, 1                  // 输出：名称数组 + 期望输出数量
    );
    
    // 提取输出数据
    float* outputData = outputTensors[0].GetTensorMutableData<float>();
    size_t outputSize = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    
    // 复制数据到 vector（因为 outputTensors 销毁后指针会失效）
    return std::vector<float>(outputData, outputData + outputSize);
}
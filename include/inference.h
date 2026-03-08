#pragma once
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <memory>    
#include <vector>    
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <memory>
#include <string> 


// Logger for TensorRT
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Only log errors and warnings to see important messages
        if (severity == Severity::kERROR || severity == Severity::kINTERNAL_ERROR || severity == Severity::kWARNING) {
            std::cout << msg << std::endl;
        }
    }
};
extern Logger gLogger;
// 前向声明 types.h 中的内容（如果需要）
struct Detection;

class Inference {
public:
    explicit Inference(const std::string& modelPath);   // 构造函数，接受模型路径参数
    ~Inference();
    // std::vector<float> run(const cv::Mat& inputBlob);   // 运行推理，接受预处理后的图像并返回原始输出数据
    std::vector<float> run(float* inputBlob_gpu); // 重载 run 方法，接受 GPU 上的输入 blob
    std::vector<int64_t> getOutputShape() const { return outputShape_; } // 获取输出张量的形状

private:

    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    
    std::vector<void*> buffers_;
    std::vector<int> bufferSizes_;
    
    std::vector<int64_t> outputShape_;
    size_t outputSize_;

    std::string inputTensorName_; 
    std::string outputTensorName_;
    int inputBufferIndex_ = -1;   // 记录输入 buffer 的索引
    int outputBufferIndex_ = -1;  // 记录输出 buffer 的索引
    cudaStream_t stream_ = nullptr;             // CUDA 流
    // bool initialized_ = false;   // 标记是否成功初始化
};
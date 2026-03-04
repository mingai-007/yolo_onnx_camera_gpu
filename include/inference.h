#pragma once
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

// 前向声明 types.h 中的内容（如果需要）
struct Detection;

class Inference {
public:
    explicit Inference(const std::string& modelPath);   // 构造函数，接受模型路径参数
    // ~Inference();             // 默认析构函数，智能指针会自动清理资源
    std::vector<float> run(const cv::Mat& inputBlob);   // 运行推理，接受预处理后的图像并返回原始输出数据
    std::vector<int64_t> getOutputShape() const { return outputShape_; } // 获取输出张量的形状

private:
    Ort::Env env_;
    
    // ✅ 使用 unique_ptr 管理 Session，因为 Session 没有默认构造函数
    std::unique_ptr<Ort::Session> session_;
    
    // ✅ 使用智能指针管理字符串生命周期（关键！）
    std::unique_ptr<char[]> inputNamePtr_;
    std::unique_ptr<char[]> outputNamePtr_;
    
    std::string inputNameStr_;       // 用于 Run() 的字符串
    std::string outputNameStr_;      // 用于 Run() 的字符串
    
    std::vector<int64_t> outputShape_;
    
    // 保存 allocator 引用（可选）
    Ort::AllocatorWithDefaultOptions allocator_;
};
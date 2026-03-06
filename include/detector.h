#pragma once
#include <memory>
#include <vector>
#include <string>
#include "types.h"
#include "preprocessor.h"
#include "inference.h"
#include "postprocessor.h"
#include "visualizer.h"

class Detector {
public:
    explicit Detector(const ModelConfig& config);           // 构造函数，接受模型配置参数
    std::vector<Detection> detect(const cv::Mat& image);    // 检测函数，接受原始图像并返回检测结果列表
    void drawResults(cv::Mat& image, const std::vector<Detection>& detections);

private:
    std::unique_ptr<PreProcessor> preProcessor_;        // 预处理器
    std::unique_ptr<Inference> inference_;              // 推理器
    std::unique_ptr<PostProcessor> postProcessor_;      // 后处理器
    std::unique_ptr<Visualizer> visualizer_;            // 可视化器
    std::vector<std::string> classes_;                  // 类别名称列表
};
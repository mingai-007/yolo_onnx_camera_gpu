#pragma once
#include <opencv2/opencv.hpp>
#include "types.h"

class PreProcessor {
public:
    // 构造函数，接受模型输入图像的宽度和高度
    PreProcessor(int inputWidth, int inputHeight);

    // 预处理函数，接受原始图像并返回预处理后的图像
    cv::Mat process(const cv::Mat& image);

    // 获取缩放因子，供后续将检测结果映射回原图坐标使用
    float getScaleX() const { return scaleX_; }
    float getScaleY() const { return scaleY_; }

private:
    int inputWidth_;        // 模型输入图像的宽度
    int inputHeight_;       // 模型输入图像的高度
    float scaleX_;          // 水平缩放因子，用于将检测结果映射回原图坐标
    float scaleY_;      // 垂直缩放因子，用于将检测结果映射回原图坐标
};
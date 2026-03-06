#pragma once
#include <opencv2/opencv.hpp>
#include "types.h"

struct PreprocessResult {
    cv::Mat blob;
    float scale;
    int padX;
    int padY;
};

class PreProcessor {
public:
    // 构造函数，接受模型输入图像的宽度和高度
    PreProcessor(int inputWidth, int inputHeight);

    // 预处理函数，接受原始图像并返回预处理后的图像
    PreprocessResult process(const cv::Mat& image);

private:
    int inputWidth_;        // 模型输入图像的宽度
    int inputHeight_;       // 模型输入图像的高度

};
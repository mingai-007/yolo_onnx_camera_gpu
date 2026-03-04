// types.h

#pragma once    //防止头文件被重复包含
#include <opencv2/opencv.hpp>

struct Detection {
    cv::Rect box;       // OpenCV 的矩形类，包含 x, y, width, height
    float confidence;   // 模型对这个检测结果的确信程度
    int classId;        // 检测到的物体类别编号
};

struct ModelConfig {
    std::string modelPath;      // 模型文件路径
    float confThreshold = 0.45f;    // 置信度阈值，只有大于这个值的检测结果才会被保留
    float nmsThreshold = 0.45f;     // 非极大值抑制（NMS）阈值，用于去除重叠的检测结果
    int inputWidth = 640;       // 模型输入图像的宽度
    int inputHeight = 640;      // 模型输入图像的高度
};
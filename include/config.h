// config.h


#pragma once
#include "types.h"
#include <string>
#include <vector>

class Config {
public:
    static Config& getInstance() {
        static Config instance; 
        return instance;
    }

    ModelConfig& getModelConfig() { return modelConfig_; }  // 获取模型配置
    const std::vector<std::string>& getClasses() { return classes_; }   // 获取类别名称列表

private:
    Config() {
        initClasses();
    }

    void initClasses() {
        classes_ = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        };
    }

    ModelConfig modelConfig_;
    std::vector<std::string> classes_;
};
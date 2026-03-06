#pragma once
#include <vector>
#include "types.h"

class PostProcessor {
public:
    PostProcessor(float confThreshold, float nmsThreshold);
    std::vector<Detection> process(float* output, 
                                   const std::vector<int64_t>& shape,
                                   float scale, 
                                   float padX,
                                   float padY,
                                   int numClasses);

private:
    float confThreshold_;   // 置信度阈值，只有大于这个值的检测结果才会被保留
    float nmsThreshold_;    // 非极大值抑制（NMS）阈值，用于去除重叠的检测结果
};
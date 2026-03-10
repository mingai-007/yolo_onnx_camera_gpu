#include "postprocessor.h"
#include <algorithm>

PostProcessor::PostProcessor(float confThreshold, float nmsThreshold)
    : confThreshold_(confThreshold), nmsThreshold_(nmsThreshold) {}

std::vector<Detection> PostProcessor::process(float* output,
                                               const std::vector<int64_t>& shape,
                                               float scale,
                                               float padX,
                                               float padY,
                                               int numClasses) {
    // 适配 [1, 84, 8400] 格式 (YOLOv8/v9 标准输出)
    int numAnchors = static_cast<int>(shape[2]);
    
    std::vector<cv::Rect2f> boxes;      // 保留浮点精度用于后续计算
    std::vector<float> scores;
    std::vector<int> classIds;
    
    for (int i = 0; i < numAnchors; ++i) {
        // 查找最大类别分数
        float maxScore = 0.0f;
        int classId = 0;
        
        for (int c = 0; c < numClasses; ++c) {
            // 输出格式: [x, y, w, h, cls0_score, cls1_score, ...]
            // 每个特征点有 (4 + numClasses) 个值，按通道优先存储
            float score = output[(4 + c) * numAnchors + i];
            if (score > maxScore) {
                maxScore = score;
                classId = c;
            }
        }
        
        // 置信度过滤
        if (maxScore > confThreshold_) {
            // 解析边界框坐标 (cx, cy, w, h)
            float cx = output[0 * numAnchors + i];
            float cy = output[1 * numAnchors + i];
            float w = output[2 * numAnchors + i];
            float h = output[3 * numAnchors + i];
            
            // 转换为左上角坐标 + 宽高，并缩放到原图尺寸
            float left = ((cx - w / 2.0f) - padX) / scale;
            float top = ((cy - h / 2.0f) - padY) / scale;
            float width = w / scale;
            float height = h / scale;
            
            boxes.emplace_back(left, top, width, height);
            scores.push_back(maxScore);
            classIds.push_back(classId);
        }
    }
    
    // NMS - OpenCV 只接受 cv::Rect (int) 或 cv::Rect2d (double)
    std::vector<int> nmsIndices;
    
    // ✅ 关键修复：转换为整数矩形用于 NMS
    std::vector<cv::Rect> intBoxes;
    intBoxes.reserve(boxes.size());
    for (const auto& box : boxes) {
        intBoxes.emplace_back(
            static_cast<int>(box.x),
            static_cast<int>(box.y),
            static_cast<int>(box.width),
            static_cast<int>(box.height)
        );
    }
    
    // 调用 OpenCV NMS
    cv::dnn::NMSBoxes(intBoxes, scores, confThreshold_, nmsThreshold_, nmsIndices);
    
    // 构建最终结果（保留原始浮点框精度）
    std::vector<Detection> results;
    results.reserve(nmsIndices.size());
    
    for (int idx : nmsIndices) {
        Detection det;
        det.box = boxes[idx];           // ✅ 使用原始浮点框，保持精度
        det.confidence = scores[idx];
        det.classId = classIds[idx];
        results.push_back(det);
    }
    
    return results;
}
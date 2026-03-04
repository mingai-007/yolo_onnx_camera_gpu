#include "visualizer.h"

void Visualizer::draw(cv::Mat& image, 
                      const std::vector<Detection>& detections,
                      const std::vector<std::string>& classes) {
    for (const auto& det : detections) {
        cv::Scalar color(0, 255, 0);
        cv::rectangle(image, det.box, color, 2);
        
        std::string label = classes[det.classId] + ": " + 
                           cv::format("%.2f", det.confidence);
        cv::putText(image, label, 
                   cv::Point(det.box.x, det.box.y - 10),
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
    }
}
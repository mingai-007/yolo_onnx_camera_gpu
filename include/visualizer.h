#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include "types.h"

class Visualizer {
public:
    void draw(cv::Mat& image, 
              const std::vector<Detection>& detections,
              const std::vector<std::string>& classes);
};
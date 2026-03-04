#include "preprocessor.h"

PreProcessor::PreProcessor(int inputWidth, int inputHeight)
    : inputWidth_(inputWidth), inputHeight_(inputHeight), scaleX_(1.0f), scaleY_(1.0f) {}

cv::Mat PreProcessor::process(const cv::Mat& image) {
    cv::Mat blob;
    scaleX_ = static_cast<float>(image.cols) / inputWidth_;
    scaleY_ = static_cast<float>(image.rows) / inputHeight_;
    
    cv::dnn::blobFromImage(image, blob, 1.0 / 255.0, 
                          cv::Size(inputWidth_, inputHeight_),
                          cv::Scalar(0, 0, 0), true, false);
    return blob;
}
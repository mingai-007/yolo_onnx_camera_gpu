#include "preprocessor.h"

PreProcessor::PreProcessor(int inputWidth, int inputHeight)
    : inputWidth_(inputWidth), inputHeight_(inputHeight) {}

PreprocessResult PreProcessor::process(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Input image is empty");
    }

    float scale = std::min(inputWidth_ / (float)image.cols, 
                          inputHeight_ / (float)image.rows);
    int new_w = static_cast<int>(image.cols * scale);
    int new_h = static_cast<int>(image.rows * scale);

    int pad_x = (inputWidth_ - new_w) / 2;
    int pad_y = (inputHeight_ - new_h) / 2;
    
    cv::Mat resized, blob;
    cv::resize(image, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);
    cv::copyMakeBorder(resized, resized, pad_y, inputHeight_ - new_h - pad_y, 
                      pad_x, inputWidth_ - new_w - pad_x, 
                      cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    // 转为 blob [C,H,W] -> [1,C,H,W], BGR->RGB, /255.0
    cv::dnn::blobFromImage(resized, blob, 1.0 / 255.0, 
                          cv::Size(), cv::Scalar(), true, false, CV_32F);

    // scaleX_ = scaleY_ = scale; 

    return {blob, scale, pad_x, pad_y};  
}
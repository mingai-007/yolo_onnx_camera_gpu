#include "detector.h"
#include "config.h"

Detector::Detector(const ModelConfig& config) {
    preProcessor_ = std::make_unique<PreProcessor>(
        config.inputWidth, config.inputHeight);
    
    inference_ = std::make_unique<Inference>(config.modelPath);
    
    postProcessor_ = std::make_unique<PostProcessor>(
        config.confThreshold, config.nmsThreshold);
    
    visualizer_ = std::make_unique<Visualizer>();
    
    classes_ = Config::getInstance().getClasses();
}

std::vector<Detection> Detector::detect(const cv::Mat& image) {

    // std::cout << "Starting detection on image of size: " << image.size << std::endl;
    auto preprocessResult = preProcessor_->process(image);
    // std::cout << "Preprocessing completed. Blob shape: " << preprocessResult.blob.size << std::endl;

    auto output = inference_->run(preprocessResult.blob);
    // std::cout << "Inference completed. Output size: " << output.size() << std::endl;

    auto detections = postProcessor_->process(
        output.data(),
        inference_->getOutputShape(),
        preprocessResult.scale,
        preprocessResult.padX,
        preprocessResult.padY,
        static_cast<int>(classes_.size())
    );
    
    return detections;
}

void Detector::drawResults(cv::Mat& image, 
                           const std::vector<Detection>& detections) {
    visualizer_->draw(image, detections, classes_);
}
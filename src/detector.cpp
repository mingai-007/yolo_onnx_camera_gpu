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

    auto start_preProcessor = std::chrono::high_resolution_clock::now();

    auto preprocessResult = preProcessor_->process(image);
    const cv::Mat& blob = preprocessResult.blob;

    // std::cout << "Preprocessing completed. Blob shape: " << blob.size << std::endl;
    auto end_preProcessor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_preProcessor = end_preProcessor - start_preProcessor;
    // std::cout << "Preprocessing time: " << elapsed_preProcessor.count() << " ms" << std::endl;

    auto start_inference = std::chrono::high_resolution_clock::now();

    auto output = inference_->run(blob);

    auto end_inference = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_inference = end_inference - start_inference;
    // std::cout << "Inference time: " << elapsed_inference.count() << " ms" << std::endl;


    // std::cout << "Inference completed. Output size: " << output.size() << std::endl;
    auto start_postProcessor = std::chrono::high_resolution_clock::now();
    auto detections = postProcessor_->process(
        output.data(),
        inference_->getOutputShape(),
        preprocessResult.scale,
        preprocessResult.padX,
        preprocessResult.padY,
        static_cast<int>(classes_.size())
    );
    
    auto end_postProcessor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_postProcessor = end_postProcessor - start_postProcessor;
    // std::cout << "Postprocessing time: " << elapsed_postProcessor.count() << " ms" << std::endl;

    return detections;
}

void Detector::drawResults(cv::Mat& image, const std::vector<Detection>& detections) {
    auto start_visualizer = std::chrono::high_resolution_clock::now();
    visualizer_->draw(image, detections, classes_);
    auto end_visualizer = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_visualizer = end_visualizer - start_visualizer;
    // std::cout << "Visualization time: " << elapsed_visualizer.count() << " ms" << std::endl;
}
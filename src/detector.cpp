#include "detector.h"
#include "config.h"
#include "preprocessor_cuda.h"

// #include <condition_variable>
// #include <atomic>
// #include <thread>
// #include <mutex>

Detector::Detector(const ModelConfig& config) {

    gpuPreProcessor_ = std::make_unique<GpuPreProcessor>(
        config.inputWidth, config.inputHeight);
    
    inference_ = std::make_unique<Inference>(config.modelPath);
    
    postProcessor_ = std::make_unique<PostProcessor>(
        config.confThreshold, config.nmsThreshold);
    
    visualizer_ = std::make_unique<Visualizer>();
    
    classes_ = Config::getInstance().getClasses();
}

std::vector<Detection> Detector::detect(const cv::Mat& image) {

    auto start_preProcessor = std::chrono::high_resolution_clock::now();

    auto preprocessResult = gpuPreProcessor_->process(image);

    auto end_preProcessor = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_preProcessor = end_preProcessor - start_preProcessor;
    // std::cout << "Preprocessing time: " << elapsed_preProcessor.count() << " ms" << std::endl;

    auto start_inference = std::chrono::high_resolution_clock::now();

    auto output = inference_->run(preprocessResult.blob);

    auto end_inference = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_inference = end_inference - start_inference;
    // std::cout << "Inference time: " << elapsed_inference.count() << " ms" << std::endl;


    // std::cout << "Inference completed. Output size: " << output.size() << std::endl;
    auto start_postProcessor = std::chrono::high_resolution_clock::now();
    auto detections = postProcessor_->process(
        output.data(),
        inference_->getOutputShape(),
        preprocessResult.scale,
        preprocessResult.pad_x,
        preprocessResult.pad_y,
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


MultiThreadDetectorManager::MultiThreadDetectorManager(const ModelConfig& config, int numThreads)
    : num_threads_(numThreads) {

    if (numThreads <= 0) {  // 如果用户没有指定线程数，自动检测硬件线程数
        num_threads_ = std::thread::hardware_concurrency();
        if (num_threads_ == 0) num_threads_ = 4; // 如果无法检测到硬件线程数，默认使用4线程
    }

    // 为每个线程创建一个 Detector 实例
    for (int i = 0; i < num_threads_; ++i) {
        detectors_.emplace_back(std::make_unique<Detector>(config));
    }

    // 启动工作线程
    for (int i = 0; i < num_threads_; ++i) {
        threads_.emplace_back(&MultiThreadDetectorManager::workerThread, this, i);
    }
}

MultiThreadDetectorManager::~MultiThreadDetectorManager() {
    shutdown();
    for (auto& t : threads_) {
        t.join();
    }
}


void MultiThreadDetectorManager::submitImage(const cv::Mat& image) {
    std::unique_lock<std::mutex> lock(task_queue_mutex_);
    if (stop_) {
        return; // Ignore new tasks if shutting down
    }
    task_queue_.emplace(Task{image.clone()}); // Clone image to ensure data validity
    lock.unlock();
    task_cv_.notify_one(); // Notify one waiting worker thread
}

bool MultiThreadDetectorManager::getResult(cv::Mat& processed_image, std::vector<Detection>& detections) {
    std::unique_lock<std::mutex> lock(result_queue_mutex_);
    if (!result_queue_.empty()) {
        auto res = std::move(result_queue_.front());
        result_queue_.pop();
        lock.unlock();

        processed_image = std::move(res.processed_image);
        detections = std::move(res.detections);
        return true;
    }
    // Optional: Wait briefly if queue is empty
    // if (result_cv_.wait_for(lock, std::chrono::milliseconds(1), [this]{ return !result_queue_.empty(); })) {
    //     auto res = std::move(result_queue_.front());
    //     result_queue_.pop();
    //     processed_image = std::move(res.processed_image);
    //     detections = std::move(res.detections);
    //     return true;
    // }
    return false; // No results available
}

void MultiThreadDetectorManager::shutdown() {
    {
        std::lock_guard<std::mutex> lock(task_queue_mutex_);
        stop_ = true;
    }
    task_cv_.notify_all(); // Wake up all waiting threads
    result_cv_.notify_all(); // Wake up threads waiting on results (optional, helps cleanup)
}

void MultiThreadDetectorManager::workerThread(int thread_id) {
    // Each thread has its own Detector instance
    auto& detector = *(detectors_[thread_id]);

    while (true) {
        Task task;
        {
            std::unique_lock<std::mutex> lock(task_queue_mutex_);
            task_cv_.wait(lock, [this] { return stop_ || !task_queue_.empty(); });

            if (stop_ && task_queue_.empty()) {
                return; // Exit if stopping and no more tasks
            }

            if (!task_queue_.empty()) {
                task = std::move(task_queue_.front());
                task_queue_.pop();
            } else {
                 continue; // Spurious wake-up, go back to wait
            }
        }

        // Perform detection using this thread's Detector instance
        auto detections = detector.detect(task.image);

        // Draw results on the processed frame
        cv::Mat result_frame = task.image.clone(); // Start with the original image
        detector.drawResults(result_frame, detections);

        // Put the result into the result queue
        std::lock_guard<std::mutex> lock(result_queue_mutex_);
        result_queue_.emplace(Result{std::move(result_frame), std::move(detections)});
        
        result_cv_.notify_one(); // Notify the main thread that a result is ready
    }
}
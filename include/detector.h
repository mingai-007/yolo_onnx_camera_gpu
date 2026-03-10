#pragma once
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "types.h"
// #include "preprocessor.h"
#include "preprocessor_cuda.h"
#include "inference.h"
#include "postprocessor.h"
#include "visualizer.h"

class Detector {
public:
    explicit Detector(const ModelConfig& config);           // 构造函数，接受模型配置参数
    std::vector<Detection> detect(const cv::Mat& image);    // 检测函数，接受原始图像并返回检测结果列表
    void drawResults(cv::Mat& image, const std::vector<Detection>& detections);

private:
    // std::unique_ptr<PreProcessor> preProcessor_;        // 预处理器
    std::unique_ptr<GpuPreProcessor> gpuPreProcessor_;  // GPU 预处理器（可选）
    std::unique_ptr<Inference> inference_;              // 推理器
    std::unique_ptr<PostProcessor> postProcessor_;      // 后处理器
    std::unique_ptr<Visualizer> visualizer_;            // 可视化器
    std::vector<std::string> classes_;                  // 类别名称列表
};

class MultiThreadDetectorManager {
public:
    explicit MultiThreadDetectorManager(const ModelConfig& config, int numThreads);
    ~MultiThreadDetectorManager();

    void submitImage(const cv::Mat& image);

    bool getResult(cv::Mat& processed_image, std::vector<Detection>& detections);

    void shutdown();
private:
    struct Task
    {
        cv::Mat image;
    };

    struct Result
    {
        cv::Mat processed_image;
        std::vector<Detection> detections;
    };
    
    void workerThread(int thread_id);

    std::vector<std::unique_ptr<Detector>> detectors_;
    std::vector<std::thread> threads_;
    std::queue<Task> task_queue_;
    std::queue<Result> result_queue_;
    mutable std::mutex task_queue_mutex_;
    mutable std::mutex result_queue_mutex_;
    std::condition_variable task_cv_;
    std::condition_variable result_cv_;
    std::atomic<bool> stop_{false};
    int num_threads_;
    
};
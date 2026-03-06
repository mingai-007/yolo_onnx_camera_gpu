#include <iostream>
#include <chrono> 
#include "detector.h"
#include "config.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <model.onnx> <camera_id>" << std::endl;
        std::cerr << "  camera_id: 0 (default), 1, 2, ... or video file path" << std::endl;
        return -1;
    }

    try {
        // 配置模型
        std::cout <<"begin config..."<<std::endl;
        ModelConfig config;
        config.modelPath = argv[1];
        config.confThreshold = 0.45f;
        config.nmsThreshold = 0.45f;
        config.inputWidth = 640;
        config.inputHeight = 640;
        
        Config::getInstance().getModelConfig() = config;    // 将模型配置保存到全局配置单例中
        
        // 创建检测器
        std::cout <<"begin create detector..."<<std::endl;
        Detector detector(config);
        
        // 打开摄像头
        std::cout <<"begin open camera..."<<std::endl;
        int cameraId = std::stoi(argv[2]);
        cv::VideoCapture cap(cameraId);
        if (!cap.isOpened()) {
            std::cerr << "Failed to open camera or video: " << argv[2] << std::endl;
            return -1;
        } else {
            std::cout << "Camera/video opened successfully: " << argv[2] << std::endl;
        }

        cv::Mat frame;
        int frameCount = 0;
        double totalFps =0.0;
        while (true)
        {
           cap>>frame;
           if(frame.empty()){
                std::cout<<"End of video stream or failed to capture frame."<<std::endl;
                break;
           }
        
            auto start = std::chrono::high_resolution_clock::now();
            
            auto detections = detector.detect(frame);
            detector.drawResults(frame, detections);

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> elapsed = end - start;
            double currentFps = 1000.0 / elapsed.count();
            totalFps = (frameCount * totalFps + currentFps) / (frameCount + 1);
            frameCount++;

            

            std::string fpsText = cv::format("FPS: %.1f | Objects: %zu", currentFps, detections.size());
            cv::putText(frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

            cv::imshow("Real-time Detection", frame);

            // cv::waitKey(1);

            if (cv::waitKey(1) == 'q') { // 等待1ms，按 'q' 键退出循环
                break;
            }

        }
        
        cap.release();
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        cv::destroyAllWindows();
        return -1;
    }
    
    return 0;
}
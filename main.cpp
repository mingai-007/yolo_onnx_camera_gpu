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
        
        int numThreads = 4;
        std::cout << "Creating MultiThreadDetectorManager with " << numThreads << " threads..."<<std::endl;
        MultiThreadDetectorManager detector_manager(config, numThreads);

        // 打开摄像头
        std::cout <<"begin open camera..."<<std::endl;
        int cameraId = std::stoi(argv[2]);
        cv::VideoCapture cap(cameraId, cv::CAP_V4L2); // 尝试打开摄像头
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280); // 设置宽度
	    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720); // 设置高度
	    cap.set(cv::CAP_PROP_FPS, 30); // 设置帧率
	    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M','J','P','G'));	// 设置视频编码格式为MJPEG
        if (!cap.isOpened()) {
            std::cerr << "Failed to open camera or video: " << argv[2] << std::endl;
            return -1;
        } else {
            std::cout << "Camera/video opened successfully: " << argv[2] << std::endl;
        }

        cv::Mat frame;
        int frameCount = 0;
        auto program_start_time = std::chrono::high_resolution_clock::now(); // Record start time
        std::chrono::high_resolution_clock::time_point last_display_time = std::chrono::high_resolution_clock::now();
        double instantaneous_fps = 0.0;
        double average_fps = 0.0;
        
        std::cout << "Starting real-time detection loop. Press 'q' to quit." << std::endl;

        auto inference_start_time = std::chrono::high_resolution_clock::now();

        while (true)
        {
           cap>>frame;
           if(frame.empty()){
                std::cout<<"End of video stream or failed to capture frame."<<std::endl;
                break;
           }
           
            detector_manager.submitImage(frame);
            cv::Mat processed_frame;
            std::vector<Detection> detections;

            if (detector_manager.getResult(processed_frame, detections)) {
                frameCount++;

                auto inference_end_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> inference_duration = inference_end_time - inference_start_time;

                if(inference_duration.count()>1.0){
                    average_fps = frameCount / inference_duration.count();
                    inference_start_time = inference_end_time;
                    frameCount = 0;
                }
                
                std::string fpsText = cv::format("AVG FPS: %.1f | Objects: %zu", average_fps, detections.size());
                cv::putText(processed_frame, fpsText, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);

                cv::imshow("Real-time Detection", processed_frame);

                int key = cv::waitKey(1);
                 if (key == 'q') {
                    break;
                }
            }
        }
        detector_manager.shutdown();
        cap.release();
        cv::destroyAllWindows();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        cv::destroyAllWindows();
        return -1;
    }
    
    return 0;
}
# yolo_onnx_cpp_camera_gpu
在Ubuntu上利用yolo进行实时目标检测  
项目采用C++版本，推理阶段启用了TensorRT加速  
预处理是在gpu  

OpenCV version: 4.6.0  
CUDA version: 12.0.140  

|    Source   |     Size    | Model       | FPS   | GPU Usage (%) |
|-------------|-------------|-------------|-------|---------------|
|    Camera   |  1280*720   | yolo11.onnx |   30  |    66-70      |
|   Video     |  1280*720   | yolo11.onnx |       |               |
|   Video     |  1920*1080  | yolo11.onnx |   40  |      93       |
|   Video     |  3840*2160  | yolo11.onnx |   35  |     88-93     |
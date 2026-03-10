# yolo_onnx_cpp_camera_gpu
在Ubuntu上利用yolo进行实时目标检测  
项目采用C++版本，推理阶段启用了TensorRT加速  
预处理还是在cpu 
OpenCV version: 4.6.0  
CUDA version: 12.0.140  

| Model       | FPS   | GPU Usage (%) |
|-------------|-------|---------------|
| yolo11.onnx |   27  |    65         |
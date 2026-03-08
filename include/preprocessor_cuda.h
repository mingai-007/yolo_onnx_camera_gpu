#pragma once
#include <opencv2/opencv.hpp>
#include <npp.h>
#include <cuda_runtime.h>

struct GpuPreprocessResult {
    float* blob;              // GPU 指针：[3,H,W] float
    float scale;
    int pad_x, pad_y;
};


class GpuPreProcessor
{
public:
    GpuPreProcessor(int input_w, int input_h);
    ~GpuPreProcessor();
    GpuPreprocessResult process(const cv::Mat& h_image);

private:

    void launchPreprocessKernel(const uint8_t* src, int src_w, int src_h, 
                                float* dst, int dst_w, int dst_h,
                                float scale, int pad_x, int pad_y,
                                cudaStream_t stream);
    int input_w_, input_h_;
    uint8_t *d_input_;
    float* d_blob_;
    cudaStream_t stream_;
};



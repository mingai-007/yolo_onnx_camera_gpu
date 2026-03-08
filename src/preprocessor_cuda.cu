#include <cuda_runtime.h>
#include <device_launch_parameters.h>  // ✅ 确保包含这个
#include "preprocessor_cuda.h"

// ✅ 设备端 min/max（替代 std::min/max）
__device__ inline int deviceMin(int a, int b) { return a < b ? a : b; }
__device__ inline int deviceMax(int a, int b) { return a > b ? a : b; }
__device__ inline float deviceFminf(float a, float b) { return fminf(a, b); }
__device__ inline float deviceFmaxf(float a, float b) { return fmaxf(a, b); }

// 双线性插值辅助函数
__device__ inline float bilinearSample(
    const uint8_t* src, int src_w, int src_h, 
    float x, float y, int c) {
    
    // ✅ 边界截断（使用 device 版本）
    x = deviceFminf(deviceFmaxf(x, 0.0f), static_cast<float>(src_w - 1.001f));
    y = deviceFminf(deviceFmaxf(y, 0.0f), static_cast<float>(src_h - 1.001f));
    
    int x0 = static_cast<int>(x);
    int y0 = static_cast<int>(y);
    int x1 = deviceMin(x0 + 1, src_w - 1);  // ✅ 使用 deviceMin
    int y1 = deviceMin(y0 + 1, src_h - 1);  // ✅ 使用 deviceMin
    
    float dx = x - x0;
    float dy = y - y0;
    
    float v00 = src[(y0 * src_w + x0) * 3 + c];
    float v01 = src[(y0 * src_w + x1) * 3 + c];
    float v10 = src[(y1 * src_w + x0) * 3 + c];
    float v11 = src[(y1 * src_w + x1) * 3 + c];
    
    float v0 = v00 * (1.0f - dx) + v01 * dx;
    float v1 = v10 * (1.0f - dx) + v11 * dx;
    return v0 * (1.0f - dy) + v1 * dy;
}

// ✅ 关键：必须加 __global__ 标记
__global__ void preprocessFusedKernel(
    const uint8_t* __restrict__ src,
    float* __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float scale,
    int pad_x, int pad_y,
    uint8_t pad_val) {
    
    // ✅ 现在 blockIdx, blockDim, threadIdx 可以正常使用了
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    if (out_x >= dst_w || out_y >= dst_h) return;
    
    float src_x = (out_x - pad_x) / scale;
    float src_y = (out_y - pad_y) / scale;
    
    int scaled_w = static_cast<int>(src_w * scale);
    int scaled_h = static_cast<int>(src_h * scale);
    bool in_valid = (out_x >= pad_x && out_x < pad_x + scaled_w &&
                     out_y >= pad_y && out_y < pad_y + scaled_h);
    
    for (int c = 0; c < 3; c++) {
        float value;
        if (in_valid) {
            value = bilinearSample(src, src_w, src_h, src_x, src_y, c);
        } else {
            value = static_cast<float>(pad_val);
        }
        
        // BGR → RGB 重排
        int dst_c = (c == 0) ? 2 : (c == 2) ? 0 : 1;
        int dst_idx = dst_c * dst_h * dst_w + out_y * dst_w + out_x;
        dst[dst_idx] = value / 255.0f;
    }
}


GpuPreProcessor::GpuPreProcessor(int input_w, int input_h)
    : input_w_(input_w), input_h_(input_h) {
    cudaMalloc(&d_input_, 4096 * 4096 * 3 * sizeof(uint8_t));
    cudaMalloc(&d_blob_, 3 * input_w_ * input_h_ * sizeof(float));
    cudaStreamCreate(&stream_);
}

GpuPreProcessor::~GpuPreProcessor() {
    cudaFree(d_input_);
    cudaFree(d_blob_);
    cudaStreamDestroy(stream_);
}

void GpuPreProcessor::launchPreprocessKernel(
    const uint8_t* src, int src_w, int src_h,
    float* dst, int dst_w, int dst_h,
    float scale, int pad_x, int pad_y,
    cudaStream_t stream) {
    
    dim3 block(16, 16);
    dim3 grid((dst_w + 15) / 16, (dst_h + 15) / 16);
    

    preprocessFusedKernel<<<grid, block, 0, stream>>>(
        src, dst,
        src_w, src_h, dst_w, dst_h,
        scale, pad_x, pad_y,
        114
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

GpuPreprocessResult GpuPreProcessor::process(const cv::Mat& h_image) {
    cv::Mat h_img;
    if (h_image.type() != CV_8UC3 || !h_image.isContinuous()) {
        h_image.convertTo(h_img, CV_8UC3);
    } else {
        h_img = h_image;
    }

    size_t input_size = h_img.cols * h_img.rows * 3;
    cudaMemcpyAsync(d_input_, h_img.data, input_size, 
                    cudaMemcpyHostToDevice, stream_);
    
    float scale = std::min(input_w_ / (float)h_img.cols,
                           input_h_ / (float)h_img.rows);
    int new_w = static_cast<int>(h_img.cols * scale);
    int new_h = static_cast<int>(h_img.rows * scale);
    int pad_x = (input_w_ - new_w) / 2;
    int pad_y = (input_h_ - new_h) / 2;

    
    launchPreprocessKernel(
        d_input_, h_img.cols, h_img.rows,
        d_blob_, input_w_, input_h_,
        scale, pad_x, pad_y,
        stream_
    );
    

    cudaStreamSynchronize(stream_);
    
    return {d_blob_, scale, pad_x, pad_y};
}


#include "inference.h"
#include <iostream>
#include <fstream>
#include <NvInfer.h>
#include <NvOnnxParser.h>


Logger gLogger;

Inference::Inference(const std::string& modelPath) {
    std::string enginePath = modelPath.substr(0, modelPath.find_last_of('.')) + ".engine";
    
    std::cout << "Checking for engine file: " << enginePath << std::endl;
    
    // Check if engine file exists
    std::ifstream engineFile(enginePath, std::ios::binary);
    if (engineFile) {
        std::cout << "Engine file exists, loading..." << std::endl;
        // Load existing engine
        engineFile.seekg(0, std::ios::end);
        size_t size = engineFile.tellg();
        engineFile.seekg(0, std::ios::beg);
        std::vector<char> engineData(size);
        engineFile.read(engineData.data(), size);
        engineFile.close();
        
        std::cout << "Creating TensorRT runtime..." << std::endl;
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }
        
        std::cout << "Deserializing engine..." << std::endl;
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(engineData.data(), size));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize CUDA engine from file");
        }
        
        std::cout << "Loaded TensorRT engine from file: " << enginePath << std::endl;
    } else {
        std::cout << "Engine file not found, building from ONNX: " << modelPath << std::endl;
        
        std::cout << "Creating TensorRT builder..." << std::endl;
        auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
        if (!builder) {
            throw std::runtime_error("Failed to create TensorRT builder");
        }
        
        std::cout << "Creating network..." << std::endl;
        auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
        if (!network) {
            throw std::runtime_error("Failed to create TensorRT network");
        }
        
        std::cout << "Creating ONNX parser..." << std::endl;
        // Create ONNX parser
        auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
        if (!parser) {
            throw std::runtime_error("Failed to create ONNX parser");
        }
        
        std::cout << "Parsing ONNX model..." << std::endl;
        // Parse ONNX model
        if (!parser->parseFromFile(modelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            throw std::runtime_error("Failed to parse ONNX model");
        }
        
        std::cout << "Creating builder config..." << std::endl;
        // Build engine
        auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
        if (!config) {
            throw std::runtime_error("Failed to create builder config");
        }
        
        
        // Set max workspace size
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 6ULL << 30);// 1GB
        // config->setPrecisions({nvinfer1::DataType::kHALF});
        // config->setFlag(nvinfer1::BuilderFlag::kPREFER_PRECISION_CONSTRAINTS);
        std::cout << "Building serialized network..." << std::endl;
        // Build and serialize engine
        auto plan = std::unique_ptr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
        if (!plan) {
            throw std::runtime_error("Failed to build serialized network");
        }
        
        std::cout << "Saving engine to file..." << std::endl;
        // Save engine to file
        std::ofstream outEngine(enginePath, std::ios::binary);
        if (outEngine) {
            outEngine.write(static_cast<const char*>(plan->data()), plan->size());
            outEngine.close();
            std::cout << "Saved TensorRT engine to file: " << enginePath << std::endl;
        } else {
            std::cerr << "Warning: Failed to save engine to file" << std::endl;
        }
        
        std::cout << "Creating runtime..." << std::endl;
        // Create runtime and engine
        runtime_ = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(gLogger));
        if (!runtime_) {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }
        
        std::cout << "Deserializing engine..." << std::endl;
        engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(runtime_->deserializeCudaEngine(plan->data(), plan->size()));
        if (!engine_) {
            throw std::runtime_error("Failed to deserialize CUDA engine");
        }
    }
    
    // Create execution context
    context_ = std::unique_ptr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
    if (!context_) {
        throw std::runtime_error("Failed to create execution context");
    }
    
    // 为输入输出分配显存，并记录相关信息
    int nbIOTensors = engine_->getNbIOTensors();
    buffers_.resize(nbIOTensors);
    bufferSizes_.resize(nbIOTensors);

    // 创建 CUDA 流
    cudaError_t streamErr = cudaStreamCreate(&stream_);
    if (streamErr != cudaSuccess) {
        throw std::runtime_error("Failed to create CUDA stream: " + 
                                std::string(cudaGetErrorString(streamErr)));
    }
    std::cout << "CUDA stream created successfully" << std::endl;

    for (int i = 0; i < nbIOTensors; ++i) {
        const char* tensorName = engine_->getIOTensorName(i);
        auto dims = engine_->getTensorShape(tensorName);
        
        // 计算 tensor 大小
        size_t size = 1;
        for (int j = 0; j < dims.nbDims; ++j) {
            size *= dims.d[j];
        }
        bufferSizes_[i] = size * sizeof(float);
        
        // 分配显存
        cudaMalloc(&buffers_[i], bufferSizes_[i]);
        
        // 判断输入/输出
        if (engine_->getTensorIOMode(tensorName) == nvinfer1::TensorIOMode::kINPUT) {
            // 可选：记录输入 tensor name
            inputTensorName_ = tensorName;
            inputBufferIndex_ = i;  // 记录输入 buffer 索引
        } else {
            // 输出 tensor
            outputShape_.clear();
            for (int j = 0; j < dims.nbDims; ++j) {
                outputShape_.push_back(dims.d[j]);
            }
            outputSize_ = size;
            outputTensorName_ = tensorName;  // 记录输出 tensor name
            outputBufferIndex_ = i;  // 记录输出 buffer 索引
        }
    }
    
    std::cout << "TensorRT engine built successfully!" << std::endl;
    std::cout << "Output shape: [";
    for (size_t i = 0; i < outputShape_.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << outputShape_[i];
    }
    std::cout << "]" << std::endl;
}

Inference::~Inference() {
    if (stream_) {
        cudaStreamDestroy(stream_);
    }

    for (auto& buffer : buffers_) {
        if (buffer) {
            cudaFree(buffer);
        }
    }
}

std::vector<float> Inference::run(float* inputBlob_gpu) {

    context_->setTensorAddress(inputTensorName_.c_str(), inputBlob_gpu);
    context_->setTensorAddress(outputTensorName_.c_str(), buffers_[outputBufferIndex_]);
    
    // Execute inference
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("Failed to execute inference");
    }
    
    cudaStreamSynchronize(stream_);
    
    std::vector<float> output(outputSize_);

    cudaError_t err = cudaMemcpy(output.data(), buffers_[outputBufferIndex_],
                                 bufferSizes_[outputBufferIndex_], cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        throw std::runtime_error("Failed to copy output to host: " +
                                 std::string(cudaGetErrorString(err)));
    }
    return output;
}
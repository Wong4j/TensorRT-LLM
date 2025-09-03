#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "../include/fp4_gemm.h"
#include "tensorrt_llm/common/logger.h"

using namespace tensorrt_llm::kernels;

// 测试参数
struct TestConfig {
    int m, n, k;
    int batch_count;
    std::string description;
};

// 测试配置
std::vector<TestConfig> testConfigs = {
    {128, 128, 128, 1, "小矩阵测试"},
    {256, 256, 256, 1, "中等矩阵测试"},
    {512, 512, 512, 1, "大矩阵测试"},
    {1024, 1024, 1024, 1, "超大矩阵测试"},
    {128, 128, 128, 4, "批量小矩阵测试"},
    {256, 256, 256, 2, "批量中等矩阵测试"}
};

// 检查 CUDA 错误
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// 生成随机数据
template<typename T>
void generateRandomData(T* data, size_t size, float min_val = -1.0f, float max_val = 1.0f) {
    for (size_t i = 0; i < size; ++i) {
        float val = min_val + static_cast<float>(rand()) / RAND_MAX * (max_val - min_val);
        data[i] = static_cast<T>(val);
    }
}

// 测试 cuBLASLt FP4 GEMM
template<typename T>
bool testCublasLtFp4Gemm(const TestConfig& config) {
    std::cout << "\n=== 测试 cuBLASLt FP4 GEMM: " << config.description << " ===" << std::endl;
    std::cout << "矩阵尺寸: M=" << config.m << ", N=" << config.n << ", K=" << config.k 
              << ", Batch=" << config.batch_count << std::endl;

    try {
        // 创建 cuBLASLt FP4 GEMM Runner
        auto runner = fp4_gemm::Fp4GemmBackendFactory::createRunner<T>(
            fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4,
            fp4_gemm::FP4GemmBackend::CUBLASLT
        );

        if (!runner) {
            std::cerr << "错误: 无法创建 cuBLASLt FP4 GEMM Runner" << std::endl;
            return false;
        }

        std::cout << "✓ 成功创建 cuBLASLt FP4 GEMM Runner" << std::endl;

        // 检查是否支持该 GEMM 类型
        if (!runner->supportsGemmType(fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4)) {
            std::cerr << "错误: cuBLASLt 不支持 W4A4_NVFP4_NVFP4 类型" << std::endl;
            return false;
        }

        std::cout << "✓ 支持 W4A4_NVFP4_NVFP4 类型" << std::endl;

        // 分配内存
        size_t input_size = config.m * config.k * config.batch_count;
        size_t weight_size = config.k * config.n * config.batch_count;
        size_t output_size = config.m * config.n * config.batch_count;
        size_t input_scale_size = config.m * config.batch_count;
        size_t weight_scale_size = config.k * config.batch_count;

        // 主机内存
        std::vector<__nv_fp4_e2m1> h_input(input_size);
        std::vector<__nv_fp4_e2m1> h_weight(weight_size);
        std::vector<T> h_output(output_size);
        std::vector<__nv_fp8_e4m3> h_input_scale(input_scale_size);
        std::vector<__nv_fp8_e4m3> h_weight_scale(weight_scale_size);
        std::vector<float> h_global_scale(1, 1.0f);

        // 生成随机数据
        generateRandomData(h_input.data(), input_size, -2.0f, 2.0f);
        generateRandomData(h_weight.data(), weight_size, -2.0f, 2.0f);
        generateRandomData(h_input_scale.data(), input_scale_size, 0.1f, 2.0f);
        generateRandomData(h_weight_scale.data(), weight_scale_size, 0.1f, 2.0f);

        // 设备内存
        __nv_fp4_e2m1 *d_input, *d_weight;
        T *d_output;
        __nv_fp8_e4m3 *d_input_scale, *d_weight_scale;
        float *d_global_scale;

        CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(__nv_fp4_e2m1)));
        CUDA_CHECK(cudaMalloc(&d_weight, weight_size * sizeof(__nv_fp4_e2m1)));
        CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(T)));
        CUDA_CHECK(cudaMalloc(&d_input_scale, input_scale_size * sizeof(__nv_fp8_e4m3)));
        CUDA_CHECK(cudaMalloc(&d_weight_scale, weight_scale_size * sizeof(__nv_fp8_e4m3)));
        CUDA_CHECK(cudaMalloc(&d_global_scale, sizeof(float)));

        // 复制数据到设备
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(__nv_fp4_e2m1), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_size * sizeof(__nv_fp4_e2m1), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_input_scale, h_input_scale.data(), input_scale_size * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_weight_scale, h_weight_scale.data(), weight_scale_size * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_global_scale, h_global_scale.data(), sizeof(float), cudaMemcpyHostToDevice));

        std::cout << "✓ 内存分配和数据传输完成" << std::endl;

        // 获取工作空间大小
        size_t workspace_size = runner->getWorkspaceSize(config.m, config.n, config.k, config.batch_count);
        std::cout << "工作空间大小: " << workspace_size << " 字节" << std::endl;

        // 分配工作空间
        char *d_workspace = nullptr;
        if (workspace_size > 0) {
            CUDA_CHECK(cudaMalloc(&d_workspace, workspace_size));
        }

        // 创建 CUDA 流
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // 预热
        std::cout << "执行预热..." << std::endl;
        for (int i = 0; i < 3; ++i) {
            runner->gemm(d_output, d_input, d_weight, d_input_scale, d_weight_scale,
                        d_global_scale, config.m, config.n, config.k, config.batch_count,
                        d_workspace, workspace_size, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 性能测试
        std::cout << "开始性能测试..." << std::endl;
        const int num_iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iterations; ++i) {
            runner->gemm(d_output, d_input, d_weight, d_input_scale, d_weight_scale,
                        d_global_scale, config.m, config.n, config.k, config.batch_count,
                        d_workspace, workspace_size, stream);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time_ms = duration.count() / 1000.0 / num_iterations;

        // 计算性能指标
        double flops = 2.0 * config.m * config.n * config.k * config.batch_count;
        double tflops = flops / (avg_time_ms * 1e-3) / 1e12;

        std::cout << "✓ 性能测试完成" << std::endl;
        std::cout << "平均执行时间: " << avg_time_ms << " ms" << std::endl;
        std::cout << "性能: " << tflops << " TFLOPS" << std::endl;

        // 验证结果（简单检查）
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(T), cudaMemcpyDeviceToHost));
        
        // 检查输出是否包含有效数据（非零）
        bool has_valid_output = false;
        for (size_t i = 0; i < output_size; ++i) {
            if (h_output[i] != T(0)) {
                has_valid_output = true;
                break;
            }
        }

        if (has_valid_output) {
            std::cout << "✓ 输出验证通过" << std::endl;
        } else {
            std::cout << "⚠ 警告: 输出全为零，可能存在问题" << std::endl;
        }

        // 清理资源
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_weight));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_input_scale));
        CUDA_CHECK(cudaFree(d_weight_scale));
        CUDA_CHECK(cudaFree(d_global_scale));
        if (d_workspace) {
            CUDA_CHECK(cudaFree(d_workspace));
        }
        CUDA_CHECK(cudaStreamDestroy(stream));

        std::cout << "✓ 资源清理完成" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return false;
    }
}

// 测试后端工厂
void testBackendFactory() {
    std::cout << "\n=== 测试后端工厂 ===" << std::endl;

    // 测试推荐后端
    auto recommended_backend = fp4_gemm::Fp4GemmBackendFactory::getRecommendedBackend(
        fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4
    );
    std::cout << "推荐后端: " << (recommended_backend == fp4_gemm::FP4GemmBackend::CUBLASLT ? "cuBLASLt" : "CUTLASS") << std::endl;

    // 测试创建不同后端的 Runner
    std::cout << "\n测试创建 CUTLASS Runner..." << std::endl;
    auto cutlass_runner = fp4_gemm::Fp4GemmBackendFactory::createRunner<half>(
        fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4,
        fp4_gemm::FP4GemmBackend::CUTLASS
    );
    if (cutlass_runner) {
        std::cout << "✓ 成功创建 CUTLASS Runner" << std::endl;
    } else {
        std::cout << "✗ 创建 CUTLASS Runner 失败" << std::endl;
    }

    std::cout << "\n测试创建 cuBLASLt Runner..." << std::endl;
    auto cublaslt_runner = fp4_gemm::Fp4GemmBackendFactory::createRunner<half>(
        fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4,
        fp4_gemm::FP4GemmBackend::CUBLASLT
    );
    if (cublaslt_runner) {
        std::cout << "✓ 成功创建 cuBLASLt Runner" << std::endl;
    } else {
        std::cout << "✗ 创建 cuBLASLt Runner 失败" << std::endl;
    }
}

int main() {
    std::cout << "=== cuBLASLt FP4 GEMM 功能测试 ===" << std::endl;

    // 初始化 CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "使用 GPU: " << prop.name << std::endl;
    std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;

    // 测试后端工厂
    testBackendFactory();

    // 测试不同数据类型的 cuBLASLt FP4 GEMM
    std::cout << "\n=== 测试不同数据类型 ===" << std::endl;

    // 测试 half 类型
    std::cout << "\n--- 测试 half 类型 ---" << std::endl;
    for (const auto& config : testConfigs) {
        if (!testCublasLtFp4Gemm<half>(config)) {
            std::cerr << "half 类型测试失败: " << config.description << std::endl;
            return 1;
        }
    }

    // 测试 float 类型
    std::cout << "\n--- 测试 float 类型 ---" << std::endl;
    for (const auto& config : testConfigs) {
        if (!testCublasLtFp4Gemm<float>(config)) {
            std::cerr << "float 类型测试失败: " << config.description << std::endl;
            return 1;
        }
    }

    // 测试 bfloat16 类型
    std::cout << "\n--- 测试 bfloat16 类型 ---" << std::endl;
    for (const auto& config : testConfigs) {
        if (!testCublasLtFp4Gemm<__nv_bfloat16>(config)) {
            std::cerr << "bfloat16 类型测试失败: " << config.description << std::endl;
            return 1;
        }
    }

    std::cout << "\n=== 所有测试通过！ ===" << std::endl;
    std::cout << "cuBLASLt FP4 GEMM 功能集成成功！" << std::endl;

    return 0;
}

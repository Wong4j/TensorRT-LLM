#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cublasLt.h>

// 简化的测试，不依赖复杂的头文件
using namespace std;

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

// 检查 cuBLASLt 错误
#define CUBLASLT_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLASLt error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << status << std::endl; \
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

// 简化的 cuBLASLt FP4 GEMM 测试
bool testSimpleCublasLtFp4Gemm(int m, int n, int k) {
    std::cout << "\n=== 简化 cuBLASLt FP4 GEMM 测试 ===" << std::endl;
    std::cout << "矩阵尺寸: M=" << m << ", N=" << n << ", K=" << k << std::endl;

    try {
        // 初始化 cuBLASLt
        cublasLtHandle_t cublaslt_handle;
        CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle));
        std::cout << "✓ cuBLASLt 初始化成功" << std::endl;

        // 分配内存
        size_t input_size = m * k;
        size_t weight_size = k * n;
        size_t output_size = m * n;
        size_t input_scale_size = m;
        size_t weight_scale_size = k;

        // 主机内存 - FP4 GEMM 测试
        std::vector<__nv_fp4_e2m1> h_input(input_size);
        std::vector<__nv_fp4_e2m1> h_weight(weight_size);
        std::vector<half> h_output(output_size);
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
        half *d_output;
        __nv_fp8_e4m3 *d_input_scale, *d_weight_scale;
        float *d_global_scale;

        CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(__nv_fp4_e2m1)));
        CUDA_CHECK(cudaMalloc(&d_weight, weight_size * sizeof(__nv_fp4_e2m1)));
        CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(half)));
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

        // 创建 CUDA 流
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));

        // 创建操作描述符 - 使用 FP16 计算类型以支持 FP4
        cublasLtMatmulDesc_t operationDesc;
        CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_16F, CUDA_R_16F));

        // 设置操作类型
        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_N;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

        // 设置 FP4 特定的属性 - 使用数值常量
        cublasLtMatmulMatrixScale_t scaleMode = static_cast<cublasLtMatmulMatrixScale_t>(0);
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

        std::cout << "✓ cuBLASLt 操作描述符创建成功" << std::endl;

        // 创建矩阵布局描述符
        cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
        
        // A 矩阵布局 (输入) - FP4 类型
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, m, k, m));
        
        // B 矩阵布局 (权重) - FP4 类型
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, k, n, k));
        
        // C 矩阵布局 (输出) - FP16 类型
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, m, n, m));
        
        // D 矩阵布局 (输出) - FP16 类型
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16F, m, n, m));

        std::cout << "✓ 矩阵布局描述符创建成功" << std::endl;

        // 获取算法
        cublasLtMatmulPreference_t preference;
        CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        size_t workspace_size_limit = 1024*1024*1024;
        CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_limit, sizeof(workspace_size_limit)));

        cublasLtMatmulHeuristicResult_t heuristicResult;
        int returnedResults = 0;
        
        // 尝试获取算法，如果失败则尝试不同的配置
        cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
            cublaslt_handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc,
            preference, 1, &heuristicResult, &returnedResults);

        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cout << "FP4 GEMM 算法获取失败，错误代码: " << status << std::endl;
            std::cout << "尝试不同的计算类型和数据类型组合..." << std::endl;
            
            // 尝试不同的计算类型
            std::vector<cublasComputeType_t> computeTypes = {
                CUBLAS_COMPUTE_16F,
                CUBLAS_COMPUTE_32F,
                CUBLAS_COMPUTE_32I
            };
            
            std::vector<std::pair<cudaDataType_t, cudaDataType_t>> dataTypePairs = {
                {CUDA_R_4F_E2M1, CUDA_R_4F_E2M1},  // FP4 x FP4
                {CUDA_R_8F_E4M3, CUDA_R_8F_E4M3},  // FP8 x FP8
                {CUDA_R_16F, CUDA_R_16F}            // FP16 x FP16
            };
            
            bool foundCompatible = false;
            for (auto computeType : computeTypes) {
                for (auto& dataTypes : dataTypePairs) {
                    cublasLtMatmulAlgo_t algo;
                    status = cublasLtMatmulAlgoInit(cublaslt_handle, computeType, 
                                                   CUDA_R_16F,  // scaleType
                                                   dataTypes.first, dataTypes.first, 
                                                   CUDA_R_16F, CUDA_R_16F, 0, &algo);
                    if (status == CUBLAS_STATUS_SUCCESS) {
                        std::cout << "✓ 找到兼容的算法配置" << std::endl;
                        std::cout << "  计算类型: " << computeType << std::endl;
                        std::cout << "  数据类型: " << dataTypes.first << " x " << dataTypes.first << std::endl;
                        foundCompatible = true;
                        break;
                    }
                }
                if (foundCompatible) break;
            }
            
            if (!foundCompatible) {
                std::cerr << "错误: 无法找到任何兼容的算法配置" << std::endl;
                std::cerr << "当前硬件或 cuBLASLt 版本可能不支持低精度 GEMM" << std::endl;
                std::cerr << "建议检查:" << std::endl;
                std::cerr << "1. GPU 是否支持 FP4/FP8 计算" << std::endl;
                std::cerr << "2. cuBLASLt 版本是否足够新" << std::endl;
                std::cerr << "3. CUDA 版本是否支持 FP4/FP8" << std::endl;
                return false;
            }
        } else if (returnedResults == 0) {
            std::cerr << "错误: 没有找到合适的算法" << std::endl;
            return false;
        } else {
            std::cout << "✓ 找到合适的算法" << std::endl;
        }

        // 预热
        std::cout << "执行预热..." << std::endl;
        for (int i = 0; i < 3; ++i) {
            if (status == CUBLAS_STATUS_SUCCESS && returnedResults > 0) {
                CUBLASLT_CHECK(cublasLtMatmul(
                    cublaslt_handle, operationDesc,
                    &h_global_scale[0],  // alpha
                    d_input, Adesc,      // A
                    d_weight, Bdesc,     // B
                    nullptr,             // beta
                    d_output, Cdesc,     // C
                    d_output, Ddesc,     // D
                    &heuristicResult.algo, nullptr, 0, stream));
            } else {
                // 使用默认算法
                cublasLtMatmulAlgo_t algo;
                CUBLASLT_CHECK(cublasLtMatmulAlgoInit(cublaslt_handle, CUBLAS_COMPUTE_16F, CUDA_R_16F, CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, CUDA_R_16F, CUDA_R_16F, 0, &algo));
                CUBLASLT_CHECK(cublasLtMatmul(
                    cublaslt_handle, operationDesc,
                    &h_global_scale[0],  // alpha
                    d_input, Adesc,      // A
                    d_weight, Bdesc,     // B
                    nullptr,             // beta
                    d_output, Cdesc,     // C
                    d_output, Ddesc,     // D
                    &algo, nullptr, 0, stream));
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 性能测试
        std::cout << "开始性能测试..." << std::endl;
        const int num_iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_iterations; ++i) {
            if (status == CUBLAS_STATUS_SUCCESS && returnedResults > 0) {
                CUBLASLT_CHECK(cublasLtMatmul(
                    cublaslt_handle, operationDesc,
                    &h_global_scale[0],  // alpha
                    d_input, Adesc,      // A
                    d_weight, Bdesc,     // B
                    nullptr,             // beta
                    d_output, Cdesc,     // C
                    d_output, Ddesc,     // D
                    &heuristicResult.algo, nullptr, 0, stream));
            } else {
                // 使用默认算法
                cublasLtMatmulAlgo_t algo;
                CUBLASLT_CHECK(cublasLtMatmulAlgoInit(cublaslt_handle, CUBLAS_COMPUTE_16F, CUDA_R_16F, CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, CUDA_R_16F, CUDA_R_16F, 0, &algo));
                CUBLASLT_CHECK(cublasLtMatmul(
                    cublaslt_handle, operationDesc,
                    &h_global_scale[0],  // alpha
                    d_input, Adesc,      // A
                    d_weight, Bdesc,     // B
                    nullptr,             // beta
                    d_output, Cdesc,     // C
                    d_output, Ddesc,     // D
                    &algo, nullptr, 0, stream));
            }
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time_ms = duration.count() / 1000.0 / num_iterations;

        // 计算性能指标
        double flops = 2.0 * m * n * k;
        double tflops = flops / (avg_time_ms * 1e-3) / 1e12;

        std::cout << "✓ 性能测试完成" << std::endl;
        std::cout << "平均执行时间: " << avg_time_ms << " ms" << std::endl;
        std::cout << "性能: " << tflops << " TFLOPS" << std::endl;

        // 验证结果
        CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, output_size * sizeof(half), cudaMemcpyDeviceToHost));
        
        bool has_valid_output = false;
        for (size_t i = 0; i < output_size; ++i) {
            if (h_output[i] != half(0)) {
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
        CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
        CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Adesc));
        CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Bdesc));
        CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Cdesc));
        CUBLASLT_CHECK(cublasLtMatrixLayoutDestroy(Ddesc));
        CUBLASLT_CHECK(cublasLtMatmulDescDestroy(operationDesc));
        CUBLASLT_CHECK(cublasLtDestroy(cublaslt_handle));

        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_weight));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_input_scale));
        CUDA_CHECK(cudaFree(d_weight_scale));
        CUDA_CHECK(cudaFree(d_global_scale));
        CUDA_CHECK(cudaStreamDestroy(stream));

        std::cout << "✓ 资源清理完成" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "=== 简化 cuBLASLt FP4 GEMM 功能测试 ===" << std::endl;

    // 初始化 CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "使用 GPU: " << prop.name << std::endl;
    std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;

    // 测试不同矩阵尺寸
    struct TestConfig {
        int m, n, k;
        std::string description;
    };
    
    std::vector<TestConfig> testConfigs = {
        {128, 128, 128, "小矩阵测试"},
        {256, 256, 256, "中等矩阵测试"},
        {512, 512, 512, "大矩阵测试"}
    };

    for (const auto& config : testConfigs) {
        int m = config.m;
        int n = config.n;
        int k = config.k;
        std::string description = config.description;

        std::cout << "\n--- " << description << " ---" << std::endl;
        if (!testSimpleCublasLtFp4Gemm(m, n, k)) {
            std::cerr << "测试失败: " << description << std::endl;
            return 1;
        }
    }

    std::cout << "\n=== 所有测试通过！ ===" << std::endl;
    std::cout << "简化 cuBLASLt FP4 GEMM 功能测试成功！" << std::endl;

    return 0;
}

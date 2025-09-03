#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << error << std::endl; \
            return false; \
        } \
    } while(0)

#define CUBLASLT_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLASLt error at " << __FILE__ << ":" << __LINE__ << " - " << status << std::endl; \
            return false; \
        } \
    } while(0)

// 生成随机 FP4 数据
void generateRandomFp4Data(std::vector<__nv_fp4_e2m1>& data) {
    for (auto& val : data) {
        // 生成 -1.0 到 1.0 之间的随机值
        float random_val = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        val = __float2nv_fp4_e2m1(random_val);
    }
}

// 生成随机 FP8 数据
void generateRandomFp8Data(std::vector<__nv_fp8_e4m3>& data) {
    for (auto& val : data) {
        // 生成 0.1 到 2.0 之间的随机缩放值
        float random_val = 0.1f + (static_cast<float>(rand()) / RAND_MAX) * 1.9f;
        val = __float2nv_fp8_e4m3(random_val);
    }
}

bool testCorrectFp4Gemm(int m, int n, int k) {
    std::cout << "\n=== 正确的 FP4 GEMM 测试 ===" << std::endl;
    std::cout << "矩阵尺寸: M=" << m << ", N=" << n << ", K=" << k << std::endl;
    
    // 检查 16 字节对齐要求
    if (m % 16 != 0 || n % 16 != 0 || k % 16 != 0) {
        std::cout << "警告: 矩阵维度不满足 16 字节对齐要求" << std::endl;
        std::cout << "M=" << m << " % 16 = " << (m % 16) << std::endl;
        std::cout << "N=" << n << " % 16 = " << (n % 16) << std::endl;
        std::cout << "K=" << k << " % 16 = " << (k % 16) << std::endl;
    }
    
    // 初始化 cuBLASLt
    cublasLtHandle_t cublaslt_handle;
    CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle));
    std::cout << "✓ cuBLASLt 初始化成功" << std::endl;
    
    // 创建 CUDA 流
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    // 分配主机内存
    std::vector<__nv_fp4_e2m1> h_input(m * k);
    std::vector<__nv_fp4_e2m1> h_weight(k * n);
    std::vector<float> h_output(m * n);
    std::vector<__nv_fp8_e4m3> h_input_scale(m * k / 16);  // 每16个元素一个缩放因子
    std::vector<__nv_fp8_e4m3> h_weight_scale(k * n / 16); // 每16个元素一个缩放因子
    
    // 生成随机数据
    generateRandomFp4Data(h_input);
    generateRandomFp4Data(h_weight);
    generateRandomFp8Data(h_input_scale);
    generateRandomFp8Data(h_weight_scale);
    
    // 分配设备内存
    __nv_fp4_e2m1 *d_input, *d_weight;
    float *d_output;
    __nv_fp8_e4m3 *d_input_scale, *d_weight_scale;
    
    CUDA_CHECK(cudaMalloc(&d_input, h_input.size() * sizeof(__nv_fp4_e2m1)));
    CUDA_CHECK(cudaMalloc(&d_weight, h_weight.size() * sizeof(__nv_fp4_e2m1)));
    CUDA_CHECK(cudaMalloc(&d_output, h_output.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_input_scale, h_input_scale.size() * sizeof(__nv_fp8_e4m3)));
    CUDA_CHECK(cudaMalloc(&d_weight_scale, h_weight_scale.size() * sizeof(__nv_fp8_e4m3)));
    
    // 传输数据到设备
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), h_input.size() * sizeof(__nv_fp4_e2m1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), h_weight.size() * sizeof(__nv_fp4_e2m1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_scale, h_input_scale.data(), h_input_scale.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight_scale, h_weight_scale.data(), h_weight_scale.size() * sizeof(__nv_fp8_e4m3), cudaMemcpyHostToDevice));
    
    std::cout << "✓ 内存分配和数据传输完成" << std::endl;
    
    // 创建操作描述符 - 使用 CUBLAS_COMPUTE_32F
    cublasLtMatmulDesc_t operationDesc;
    CUBLASLT_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    
    // 设置缩放模式为 CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    
    std::cout << "✓ cuBLASLt 操作描述符创建成功" << std::endl;
    
    // 创建矩阵布局描述符
    // A 必须是转置的 (TN 格式)
    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc, Ddesc;
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, k, m, k));  // A^T: k x m
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, k, n, k));  // B: k x n
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_32F, m, n, m));      // C: m x n
    CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_32F, m, n, m));      // D: m x n
    
    std::cout << "✓ 矩阵布局描述符创建成功" << std::endl;
    
    // 创建算法偏好
    cublasLtMatmulPreference_t preference;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    size_t workspace_size_limit = 1024*1024*1024;
    CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size_limit, sizeof(workspace_size_limit)));
    
    // 尝试获取算法
    cublasLtMatmulHeuristicResult_t heuristicResult;
    int returnedResults = 0;
    
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        cublaslt_handle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc,
        preference, 1, &heuristicResult, &returnedResults);
    
    if (status != CUBLAS_STATUS_SUCCESS || returnedResults == 0) {
        std::cout << "FP4 GEMM 算法获取失败，错误代码: " << status << std::endl;
        std::cout << "返回结果数量: " << returnedResults << std::endl;
        
        // 尝试使用默认算法
        std::cout << "尝试使用默认算法..." << std::endl;
        cublasLtMatmulAlgo_t algo;
        status = cublasLtMatmulAlgoInit(cublaslt_handle, CUBLAS_COMPUTE_32F, 
                                       CUDA_R_32F,  // scaleType
                                       CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, 
                                       CUDA_R_32F, CUDA_R_32F, 0, &algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "错误: 无法初始化默认算法，错误代码: " << status << std::endl;
            return false;
        }
        std::cout << "✓ 使用默认算法" << std::endl;
        
        // 预热
        std::cout << "执行预热..." << std::endl;
        for (int i = 0; i < 3; ++i) {
            CUBLASLT_CHECK(cublasLtMatmul(
                cublaslt_handle, operationDesc,
                nullptr,  // alpha (使用缩放因子)
                d_input, Adesc,      // A^T
                d_weight, Bdesc,     // B
                nullptr,             // beta
                d_output, Cdesc,     // C
                d_output, Ddesc,     // D
                &algo, nullptr, 0, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // 性能测试
        std::cout << "开始性能测试..." << std::endl;
        const int num_iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            CUBLASLT_CHECK(cublasLtMatmul(
                cublaslt_handle, operationDesc,
                nullptr,  // alpha (使用缩放因子)
                d_input, Adesc,      // A^T
                d_weight, Bdesc,     // B
                nullptr,             // beta
                d_output, Cdesc,     // C
                d_output, Ddesc,     // D
                &algo, nullptr, 0, stream));
        }
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time_ms = duration.count() / 1000.0 / num_iterations;
        
        std::cout << "✓ FP4 GEMM 测试成功!" << std::endl;
        std::cout << "平均执行时间: " << avg_time_ms << " ms" << std::endl;
        std::cout << "吞吐量: " << (2.0 * m * n * k / (avg_time_ms / 1000.0) / 1e12) << " TFLOPS" << std::endl;
        
    } else {
        std::cout << "✓ 找到合适的算法" << std::endl;
        
        // 预热
        std::cout << "执行预热..." << std::endl;
        for (int i = 0; i < 3; ++i) {
            CUBLASLT_CHECK(cublasLtMatmul(
                cublaslt_handle, operationDesc,
                nullptr,  // alpha (使用缩放因子)
                d_input, Adesc,      // A^T
                d_weight, Bdesc,     // B
                nullptr,             // beta
                d_output, Cdesc,     // C
                d_output, Ddesc,     // D
                &heuristicResult.algo, nullptr, 0, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // 性能测试
        std::cout << "开始性能测试..." << std::endl;
        const int num_iterations = 10;
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            CUBLASLT_CHECK(cublasLtMatmul(
                cublaslt_handle, operationDesc,
                nullptr,  // alpha (使用缩放因子)
                d_input, Adesc,      // A^T
                d_weight, Bdesc,     // B
                nullptr,             // beta
                d_output, Cdesc,     // C
                d_output, Ddesc,     // D
                &heuristicResult.algo, nullptr, 0, stream));
        }
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double avg_time_ms = duration.count() / 1000.0 / num_iterations;
        
        std::cout << "✓ FP4 GEMM 测试成功!" << std::endl;
        std::cout << "平均执行时间: " << avg_time_ms << " ms" << std::endl;
        std::cout << "吞吐量: " << (2.0 * m * n * k / (avg_time_ms / 1000.0) / 1e12) << " TFLOPS" << std::endl;
    }
    
    // 清理资源
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input_scale));
    CUDA_CHECK(cudaFree(d_weight_scale));
    
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatrixLayoutDestroy(Ddesc);
    cublasLtMatmulDescDestroy(operationDesc);
    cublasLtMatmulPreferenceDestroy(preference);
    cublasLtDestroy(cublaslt_handle);
    CUDA_CHECK(cudaStreamDestroy(stream));
    
    return true;
}

int main() {
    std::cout << "=== 正确的 cuBLASLt FP4 GEMM 测试 ===" << std::endl;
    
    // 检查 GPU 信息
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    std::cout << "使用 GPU: " << prop.name << std::endl;
    std::cout << "计算能力: " << prop.major << "." << prop.minor << std::endl;
    
    // 测试不同的矩阵尺寸 (确保 16 字节对齐)
    std::vector<std::tuple<int, int, int>> testConfigs = {
        {128, 128, 128},   // 8x8x8 的 16 字节对齐
        {256, 256, 256},   // 16x16x16 的 16 字节对齐
        {512, 512, 512},   // 32x32x32 的 16 字节对齐
    };
    
    bool allTestsPassed = true;
    
    for (const auto& config : testConfigs) {
        int m, n, k;
        std::tie(m, n, k) = config;
        
        std::cout << "\n--- 测试配置: " << m << "x" << n << "x" << k << " ---" << std::endl;
        
        if (!testCorrectFp4Gemm(m, n, k)) {
            std::cerr << "测试失败: " << m << "x" << n << "x" << k << std::endl;
            allTestsPassed = false;
        }
    }
    
    if (allTestsPassed) {
        std::cout << "\n🎉 所有测试通过!" << std::endl;
    } else {
        std::cout << "\n❌ 部分测试失败" << std::endl;
    }
    
    return allTestsPassed ? 0 : 1;
}

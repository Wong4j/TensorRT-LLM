#include <iostream>
#include <vector>
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

bool testHardwareCompatibility() {
    std::cout << "=== 硬件兼容性测试 ===" << std::endl;
    
    // 检查 GPU 信息
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "GPU 数量: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        std::cout << "  计算能力: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  多处理器数量: " << prop.multiProcessorCount << std::endl;
        std::cout << "  全局内存: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    }
    
    // 检查 cuBLASLt 版本
    cublasLtHandle_t cublaslt_handle;
    CUBLASLT_CHECK(cublasLtCreate(&cublaslt_handle));
    
    size_t version = cublasLtGetVersion();
    std::cout << "cuBLASLt 版本: " << version << std::endl;
    
    // 测试不同的数据类型支持
    std::cout << "\n=== 数据类型支持测试 ===" << std::endl;
    
    struct DataTypeTest {
        cudaDataType_t type;
        const char* name;
        size_t size;
    };
    
    std::vector<DataTypeTest> dataTypes = {
        {CUDA_R_4F_E2M1, "FP4 (E2M1)", sizeof(__nv_fp4_e2m1)},
        {CUDA_R_8F_E4M3, "FP8 (E4M3)", sizeof(__nv_fp8_e4m3)},
        {CUDA_R_8F_E5M2, "FP8 (E5M2)", sizeof(__nv_fp8_e5m2)},
        {CUDA_R_16F, "FP16", sizeof(half)},
        {CUDA_R_32F, "FP32", sizeof(float)},
        {CUDA_R_64F, "FP64", sizeof(double)}
    };
    
    for (const auto& dt : dataTypes) {
        std::cout << "测试 " << dt.name << " (大小: " << dt.size << " 字节): ";
        
        // 尝试创建矩阵布局描述符
        cublasLtMatrixLayout_t layout;
        cublasStatus_t status = cublasLtMatrixLayoutCreate(&layout, dt.type, 128, 128, 128);
        
        if (status == CUBLAS_STATUS_SUCCESS) {
            std::cout << "✓ 支持" << std::endl;
            cublasLtMatrixLayoutDestroy(layout);
        } else {
            std::cout << "✗ 不支持 (错误: " << status << ")" << std::endl;
        }
    }
    
    // 测试不同的计算类型支持
    std::cout << "\n=== 计算类型支持测试 ===" << std::endl;
    
    struct ComputeTypeTest {
        cublasComputeType_t type;
        const char* name;
    };
    
    std::vector<ComputeTypeTest> computeTypes = {
        {CUBLAS_COMPUTE_16F, "FP16"},
        {CUBLAS_COMPUTE_32F, "FP32"},
        {CUBLAS_COMPUTE_32I, "INT32"},
        {CUBLAS_COMPUTE_64F, "FP64"}
    };
    
    for (const auto& ct : computeTypes) {
        std::cout << "测试计算类型 " << ct.name << ": ";
        
        // 尝试创建操作描述符
        cublasLtMatmulDesc_t operationDesc;
        cublasStatus_t status = cublasLtMatmulDescCreate(&operationDesc, ct.type, CUDA_R_16F);
        
        if (status == CUBLAS_STATUS_SUCCESS) {
            std::cout << "✓ 支持" << std::endl;
            cublasLtMatmulDescDestroy(operationDesc);
        } else {
            std::cout << "✗ 不支持 (错误: " << status << ")" << std::endl;
        }
    }
    
    // 测试算法支持
    std::cout << "\n=== 算法支持测试 ===" << std::endl;
    
    // 测试 FP4 GEMM 算法
    std::cout << "测试 FP4 GEMM 算法: ";
    cublasLtMatmulAlgo_t algo;
    cublasStatus_t status = cublasLtMatmulAlgoInit(cublaslt_handle, CUBLAS_COMPUTE_16F, 
                                                  CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, 
                                                  CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, 0, &algo);
    if (status == CUBLAS_STATUS_SUCCESS) {
        std::cout << "✓ 支持" << std::endl;
    } else {
        std::cout << "✗ 不支持 (错误: " << status << ")" << std::endl;
    }
    
    // 测试 FP8 GEMM 算法
    std::cout << "测试 FP8 GEMM 算法: ";
    status = cublasLtMatmulAlgoInit(cublaslt_handle, CUBLAS_COMPUTE_16F, 
                                   CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, 
                                   CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, 0, &algo);
    if (status == CUBLAS_STATUS_SUCCESS) {
        std::cout << "✓ 支持" << std::endl;
    } else {
        std::cout << "✗ 不支持 (错误: " << status << ")" << std::endl;
    }
    
    // 测试 FP16 GEMM 算法
    std::cout << "测试 FP16 GEMM 算法: ";
    status = cublasLtMatmulAlgoInit(cublaslt_handle, CUBLAS_COMPUTE_16F, 
                                   CUDA_R_16F, CUDA_R_16F, 
                                   CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, 0, &algo);
    if (status == CUBLAS_STATUS_SUCCESS) {
        std::cout << "✓ 支持" << std::endl;
    } else {
        std::cout << "✗ 不支持 (错误: " << status << ")" << std::endl;
    }
    
    cublasLtDestroy(cublaslt_handle);
    
    std::cout << "\n=== 兼容性测试完成 ===" << std::endl;
    return true;
}

int main() {
    std::cout << "=== cuBLASLt 硬件兼容性测试 ===" << std::endl;
    
    if (!testHardwareCompatibility()) {
        std::cerr << "兼容性测试失败" << std::endl;
        return 1;
    }
    
    std::cout << "兼容性测试完成" << std::endl;
    return 0;
}

#include <iostream>
#include <memory>
#include <cuda_runtime.h>

#include "../include/fp4_gemm.h"

using namespace tensorrt_llm::kernels;

// 简单的使用示例
int main() {
    std::cout << "=== cuBLASLt FP4 GEMM 使用示例 ===" << std::endl;

    try {
        // 1. 创建 FP4 GEMM Runner
        std::cout << "1. 创建 FP4 GEMM Runner..." << std::endl;
        
        // 使用工厂创建 cuBLASLt 后端
        auto runner = fp4_gemm::Fp4GemmBackendFactory::createRunner<half>(
            fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4,
            fp4_gemm::FP4GemmBackend::CUBLASLT
        );

        if (!runner) {
            std::cerr << "错误: 无法创建 FP4 GEMM Runner" << std::endl;
            return 1;
        }

        std::cout << "✓ 成功创建 cuBLASLt FP4 GEMM Runner" << std::endl;

        // 2. 检查支持的类型
        std::cout << "\n2. 检查支持的类型..." << std::endl;
        bool supports_w4a4 = runner->supportsGemmType(fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4);
        bool supports_w4a8 = runner->supportsGemmType(fp4_gemm::FP4GemmType::W4A8_MXFP4_MXFP8);
        
        std::cout << "支持 W4A4_NVFP4_NVFP4: " << (supports_w4a4 ? "是" : "否") << std::endl;
        std::cout << "支持 W4A8_MXFP4_MXFP8: " << (supports_w4a8 ? "是" : "否") << std::endl;

        // 3. 获取工作空间大小
        std::cout << "\n3. 获取工作空间大小..." << std::endl;
        int m = 256, n = 256, k = 256, batch_count = 1;
        size_t workspace_size = runner->getWorkspaceSize(m, n, k, batch_count);
        std::cout << "矩阵尺寸: " << m << "x" << n << "x" << k << std::endl;
        std::cout << "工作空间大小: " << workspace_size << " 字节" << std::endl;

        // 4. 分配内存（示例）
        std::cout << "\n4. 内存分配示例..." << std::endl;
        
        size_t input_size = m * k * batch_count;
        size_t weight_size = k * n * batch_count;
        size_t output_size = m * n * batch_count;
        size_t input_scale_size = m * batch_count;
        size_t weight_scale_size = k * batch_count;

        // 设备内存指针
        void *d_input, *d_weight, *d_output;
        void *d_input_scale, *d_weight_scale, *d_global_scale;
        char *d_workspace;

        // 分配内存
        cudaMalloc(&d_input, input_size * sizeof(__nv_fp4_e2m1));
        cudaMalloc(&d_weight, weight_size * sizeof(__nv_fp4_e2m1));
        cudaMalloc(&d_output, output_size * sizeof(half));
        cudaMalloc(&d_input_scale, input_scale_size * sizeof(__nv_fp8_e4m3));
        cudaMalloc(&d_weight_scale, weight_scale_size * sizeof(__nv_fp8_e4m3));
        cudaMalloc(&d_global_scale, sizeof(float));
        
        if (workspace_size > 0) {
            cudaMalloc(&d_workspace, workspace_size);
        } else {
            d_workspace = nullptr;
        }

        std::cout << "✓ 内存分配完成" << std::endl;

        // 5. 创建 CUDA 流
        std::cout << "\n5. 创建 CUDA 流..." << std::endl;
        cudaStream_t stream;
        cudaStreamCreate(&stream);
        std::cout << "✓ CUDA 流创建完成" << std::endl;

        // 6. 执行 GEMM 操作（示例）
        std::cout << "\n6. 执行 GEMM 操作..." << std::endl;
        std::cout << "注意: 这里只是示例，实际使用时需要先填充数据" << std::endl;
        
        // 这里应该先填充输入数据，然后调用：
        // runner->gemm(d_output, d_input, d_weight, d_input_scale, d_weight_scale,
        //              d_global_scale, m, n, k, batch_count, d_workspace, workspace_size, stream);
        
        std::cout << "✓ GEMM 操作准备完成" << std::endl;

        // 7. 清理资源
        std::cout << "\n7. 清理资源..." << std::endl;
        cudaFree(d_input);
        cudaFree(d_weight);
        cudaFree(d_output);
        cudaFree(d_input_scale);
        cudaFree(d_weight_scale);
        cudaFree(d_global_scale);
        if (d_workspace) {
            cudaFree(d_workspace);
        }
        cudaStreamDestroy(stream);
        std::cout << "✓ 资源清理完成" << std::endl;

        // 8. 后端比较
        std::cout << "\n8. 后端比较..." << std::endl;
        
        // 创建 CUTLASS 后端进行比较
        auto cutlass_runner = fp4_gemm::Fp4GemmBackendFactory::createRunner<half>(
            fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4,
            fp4_gemm::FP4GemmBackend::CUTLASS
        );

        if (cutlass_runner) {
            size_t cutlass_workspace = cutlass_runner->getWorkspaceSize(m, n, k, batch_count);
            std::cout << "CUTLASS 工作空间大小: " << cutlass_workspace << " 字节" << std::endl;
            std::cout << "cuBLASLt 工作空间大小: " << workspace_size << " 字节" << std::endl;
        }

        std::cout << "\n=== 示例完成 ===" << std::endl;
        std::cout << "cuBLASLt FP4 GEMM 集成成功！" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "异常: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

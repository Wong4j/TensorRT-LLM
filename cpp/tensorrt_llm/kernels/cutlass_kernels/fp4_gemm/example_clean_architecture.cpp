/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include "../include/fp4_gemm.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

using namespace tensorrt_llm::kernels::fp4_gemm;
using namespace tensorrt_llm::common;

// 演示新的清洁架构
void demonstrateCleanArchitecture()
{
    std::cout << "==========================================" << std::endl;
    std::cout << "演示 FP4 GEMM 清洁架构" << std::endl;
    std::cout << "==========================================" << std::endl;

    // 测试参数
    const int m = 128;
    const int n = 256;
    const int k = 512;
    const int batch_count = 1;

    // 分配 GPU 内存
    size_t A_size = m * k * sizeof(__nv_fp4_e2m1);
    size_t B_size = k * n * sizeof(__nv_fp4_e2m1);
    size_t D_size = m * n * sizeof(half);
    size_t input_sf_size = m * sizeof(__nv_fp8_e4m3);
    size_t weight_sf_size = n * sizeof(__nv_fp8_e4m3);

    void* d_A;
    void* d_B;
    void* d_D;
    void* d_input_sf;
    void* d_weight_sf;
    float* d_global_sf;

    check_cuda_error(cudaMalloc(&d_A, A_size));
    check_cuda_error(cudaMalloc(&d_B, B_size));
    check_cuda_error(cudaMalloc(&d_D, D_size));
    check_cuda_error(cudaMalloc(&d_input_sf, input_sf_size));
    check_cuda_error(cudaMalloc(&d_weight_sf, weight_sf_size));
    check_cuda_error(cudaMalloc(&d_global_sf, sizeof(float)));

    // 初始化数据
    check_cuda_error(cudaMemset(d_A, 0, A_size));
    check_cuda_error(cudaMemset(d_B, 0, B_size));
    check_cuda_error(cudaMemset(d_D, 0, D_size));
    check_cuda_error(cudaMemset(d_input_sf, 0, input_sf_size));
    check_cuda_error(cudaMemset(d_weight_sf, 0, weight_sf_size));
    
    float global_scale = 1.0f;
    check_cuda_error(cudaMemcpy(d_global_sf, &global_scale, sizeof(float), cudaMemcpyHostToDevice));

    // 创建 CUDA 流
    cudaStream_t stream;
    check_cuda_error(cudaStreamCreate(&stream));

    // 1. 使用工厂模式创建 CUTLASS 后端
    std::cout << "\n[测试 1] 使用 CUTLASS 后端..." << std::endl;
    try {
        auto cutlass_runner = Fp4GemmBackendFactory::createRunner<half>(
            FP4GemmType::W4A4_NVFP4_NVFP4, 
            FP4GemmBackend::CUTLASS);
        
        std::cout << "✅ CUTLASS 后端创建成功" << std::endl;
        std::cout << "   后端类型: " << (cutlass_runner->getBackendType() == FP4GemmBackend::CUTLASS ? "CUTLASS" : "Unknown") << std::endl;
        std::cout << "   支持 W4A4: " << (cutlass_runner->supportsGemmType(FP4GemmType::W4A4_NVFP4_NVFP4) ? "是" : "否") << std::endl;
        std::cout << "   支持 W4A8: " << (cutlass_runner->supportsGemmType(FP4GemmType::W4A8_MXFP4_MXFP8) ? "是" : "否") << std::endl;
        
        // 获取工作空间大小
        size_t workspace_size = cutlass_runner->getWorkspaceSize(m, n, k, batch_count);
        std::cout << "   工作空间大小: " << workspace_size << " 字节" << std::endl;
        
        // 分配工作空间
        char* workspace = nullptr;
        if (workspace_size > 0) {
            check_cuda_error(cudaMalloc(&workspace, workspace_size));
        }
        
        // 执行 GEMM (这里需要配置，简化示例)
        // cutlass_runner->gemm(d_D, d_A, d_B, d_input_sf, d_weight_sf, d_global_sf,
        //                     m, n, k, batch_count, config, workspace, workspace_size, stream);
        
        if (workspace) {
            check_cuda_error(cudaFree(workspace));
        }
    }
    catch (const std::exception& e) {
        std::cout << "❌ CUTLASS 后端测试失败: " << e.what() << std::endl;
    }

    // 2. 使用工厂模式创建 cuBLASLt 后端
    std::cout << "\n[测试 2] 使用 cuBLASLt 后端..." << std::endl;
    try {
        auto cublaslt_runner = Fp4GemmBackendFactory::createRunner<half>(
            FP4GemmType::W4A4_NVFP4_NVFP4, 
            FP4GemmBackend::CUBLASLT);
        
        std::cout << "✅ cuBLASLt 后端创建成功" << std::endl;
        std::cout << "   后端类型: " << (cublaslt_runner->getBackendType() == FP4GemmBackend::CUBLASLT ? "cuBLASLt" : "Unknown") << std::endl;
        std::cout << "   支持 W4A4: " << (cublaslt_runner->supportsGemmType(FP4GemmType::W4A4_NVFP4_NVFP4) ? "是" : "否") << std::endl;
        std::cout << "   支持 W4A8: " << (cublaslt_runner->supportsGemmType(FP4GemmType::W4A8_MXFP4_MXFP8) ? "是" : "否") << std::endl;
        
        // 获取工作空间大小
        size_t workspace_size = cublaslt_runner->getWorkspaceSize(m, n, k, batch_count);
        std::cout << "   工作空间大小: " << workspace_size << " 字节" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "❌ cuBLASLt 后端测试失败: " << e.what() << std::endl;
    }

    // 3. 演示自动后端选择
    std::cout << "\n[测试 3] 自动后端选择..." << std::endl;
    try {
        // 获取推荐的后端
        auto recommended_backend = Fp4GemmBackendFactory::getRecommendedBackend(FP4GemmType::W4A4_NVFP4_NVFP4);
        std::cout << "   W4A4 推荐后端: " << (recommended_backend == FP4GemmBackend::CUTLASS ? "CUTLASS" : "cuBLASLt") << std::endl;
        
        auto recommended_backend_w4a8 = Fp4GemmBackendFactory::getRecommendedBackend(FP4GemmType::W4A8_MXFP4_MXFP8);
        std::cout << "   W4A8 推荐后端: " << (recommended_backend_w4a8 == FP4GemmBackend::CUTLASS ? "CUTLASS" : "cuBLASLt") << std::endl;
        
        // 获取可用后端列表
        auto available_backends = Fp4GemmBackendFactory::getAvailableBackends();
        std::cout << "   可用后端: ";
        for (auto backend : available_backends) {
            std::cout << (backend == FP4GemmBackend::CUTLASS ? "CUTLASS " : "cuBLASLt ");
        }
        std::cout << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "❌ 自动后端选择测试失败: " << e.what() << std::endl;
    }

    // 清理资源
    check_cuda_error(cudaStreamDestroy(stream));
    check_cuda_error(cudaFree(d_A));
    check_cuda_error(cudaFree(d_B));
    check_cuda_error(cudaFree(d_D));
    check_cuda_error(cudaFree(d_input_sf));
    check_cuda_error(cudaFree(d_weight_sf));
    check_cuda_error(cudaFree(d_global_sf));

    std::cout << "\n==========================================" << std::endl;
    std::cout << "清洁架构演示完成！" << std::endl;
    std::cout << "==========================================" << std::endl;
}

int main()
{
    try {
        demonstrateCleanArchitecture();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "演示失败: " << e.what() << std::endl;
        return 1;
    }
}

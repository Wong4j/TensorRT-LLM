/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include "fp4_gemm.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/opUtils.h"
#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace tensorrt_llm
{
namespace kernels
{
namespace cublaslt_kernels
{

// 内联基类定义
class CublasLtFp4GemmRunnerInterface
{
public:
    virtual ~CublasLtFp4GemmRunnerInterface() = default;
    
    virtual void gemm(void* D, void const* A, void const* B, 
                     void const* input_sf, void const* weight_sf,
                     float const* global_sf, int m, int n, int k, 
                     int batch_count, char* workspace, const size_t workspaceBytes, 
                     cudaStream_t stream) = 0;
    
    virtual size_t getWorkspaceSize(int const m, int const n, int const k, int batch_count) = 0;
};

// 模板类定义
template <typename T>
class CublasLtFp4GemmRunner : public virtual CublasLtFp4GemmRunnerInterface
{
public:
    CublasLtFp4GemmRunner();
    ~CublasLtFp4GemmRunner();
    
    void gemm(void* D, void const* A, void const* B, 
             void const* input_sf, void const* weight_sf,
             float const* global_sf, int m, int n, int k, 
             int batch_count, char* workspace, const size_t workspaceBytes, 
             cudaStream_t stream) override;
    
    size_t getWorkspaceSize(int const m, int const n, int const k, int batch_count) override;

private:
    void validateInputTypes(void const* A, void const* B, 
                           void const* input_sf, void const* weight_sf);
    void executeCublasLtGemm(void* D, void const* A, void const* B, 
                            void const* input_sf, void const* weight_sf,
                            float const* global_sf, int m, int n, int k, 
                            char* workspace, const size_t workspaceBytes, cudaStream_t stream);
    
    cublasLtHandle_t mCublasLtHandle;
    int mSm;
};

// 模板类实现
template <typename T>
CublasLtFp4GemmRunner<T>::CublasLtFp4GemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TLLM_CUDA_CHECK(cublasLtCreate(&mCublasLtHandle));
    mSm = tensorrt_llm::common::getSMVersion();
}

template <typename T>
CublasLtFp4GemmRunner<T>::~CublasLtFp4GemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    if (mCublasLtHandle)
    {
        cublasLtDestroy(mCublasLtHandle);
    }
}

template <typename T>
void CublasLtFp4GemmRunner<T>::gemm(void* D, void const* A, void const* B,
                                    void const* input_sf, void const* weight_sf,
                                    float const* global_sf, int m, int n, int k,
                                    int batch_count, char* workspace, const size_t workspaceBytes,
                                    cudaStream_t stream)
{
    TLLM_LOG_INFO("[CublasLtFp4GemmRunner::gemm] Starting cuBLASLt FP4 GEMM");
    TLLM_LOG_INFO("[CublasLtFp4GemmRunner::gemm] GEMM dimensions: m=" + std::to_string(m) + 
                  ", n=" + std::to_string(n) + ", k=" + std::to_string(k) + 
                  ", batch_count=" + std::to_string(batch_count));
    TLLM_LOG_INFO("[CublasLtFp4GemmRunner::gemm] Workspace size: " + std::to_string(workspaceBytes) + " bytes");
    TLLM_LOG_INFO("[CublasLtFp4GemmRunner::gemm] Output type: " + std::string(typeid(T).name()));
    
    // 验证输入类型
    validateInputTypes(A, B, input_sf, weight_sf);
    
    // 执行 cuBLASLt GEMM
    executeCublasLtGemm(D, A, B, input_sf, weight_sf, global_sf, m, n, k, workspace, workspaceBytes, stream);
    
    TLLM_LOG_INFO("[CublasLtFp4GemmRunner::gemm] cuBLASLt FP4 GEMM completed successfully");
}

template <typename T>
void CublasLtFp4GemmRunner<T>::validateInputTypes(void const* A, void const* B,
                                                  void const* input_sf, void const* weight_sf)
{
    // 验证输入和权重都是 nvfp4 类型
    // 这里可以添加类型检查逻辑
    TLLM_LOG_DEBUG("Validating input types for cuBLASLt FP4 GEMM");
}

template <typename T>
void CublasLtFp4GemmRunner<T>::executeCublasLtGemm(void* D, void const* A, void const* B,
                                                   void const* input_sf, void const* weight_sf,
                                                   float const* global_sf, int m, int n, int k,
                                                   char* workspace, const size_t workspaceBytes, cudaStream_t stream)
{
    TLLM_LOG_INFO("[CublasLtFp4GemmRunner::executeCublasLtGemm] Starting cuBLASLt execution");
    TLLM_LOG_INFO("[CublasLtFp4GemmRunner::executeCublasLtGemm] Matrix dimensions: m=" + std::to_string(m) + 
                  ", n=" + std::to_string(n) + ", k=" + std::to_string(k));
    TLLM_LOG_INFO("[CublasLtFp4GemmRunner::executeCublasLtGemm] Workspace: " + std::to_string(workspaceBytes) + " bytes");
    
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    
    try
    {
        // 创建操作描述符
        TLLM_LOG_INFO("[CublasLtFp4GemmRunner::executeCublasLtGemm] Creating operation descriptor");
        TLLM_CUDA_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t transa = CUBLAS_OP_N;
        cublasOperation_t transb = CUBLAS_OP_N;
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, 
                                                     &transa, sizeof(transa)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, 
                                                     &transb, sizeof(transb)));
        
        // 设置缩放模式 - 使用默认的缩放模式
        // 注意：对于 FP4 GEMM，缩放通常通过指针设置，不需要额外的模式设置
        
        // 设置缩放指针
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, 
                                                     &input_sf, sizeof(input_sf)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, 
                                                     &weight_sf, sizeof(weight_sf)));
        
        // 创建矩阵描述符
        TLLM_LOG_INFO("[CublasLtFp4GemmRunner::executeCublasLtGemm] Creating matrix descriptors");
        TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, m, k, k));
        TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, k, n, n));
        
        // 根据输出类型创建 C 和 D 描述符
        cudaDataType_t outputType;
        if constexpr (std::is_same_v<T, half>) {
            outputType = CUDA_R_16F;
        } else if constexpr (std::is_same_v<T, float>) {
            outputType = CUDA_R_32F;
        } else if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            outputType = CUDA_R_16BF;
        } else {
            throw std::runtime_error("Unsupported output type for cuBLASLt FP4 GEMM");
        }
        TLLM_LOG_INFO("[CublasLtFp4GemmRunner::executeCublasLtGemm] Output type: " + std::to_string(outputType));
        TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, outputType, m, n, n));
        TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_4F_E2M1, m, n, n)); // D is FP4 in sample
        
        // 创建偏好描述符
        TLLM_LOG_INFO("[CublasLtFp4GemmRunner::executeCublasLtGemm] Creating preference descriptor");
        TLLM_CUDA_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        TLLM_CUDA_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                                           &workspaceBytes, sizeof(workspaceBytes)));
        
        // 获取启发式算法
        TLLM_LOG_INFO("[CublasLtFp4GemmRunner::executeCublasLtGemm] Getting heuristic algorithm");
        TLLM_CUDA_CHECK(cublasLtMatmulAlgoGetHeuristic(mCublasLtHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, 
                                                     preference, 1, &heuristicResult, &returnedResults));
        
        if (returnedResults == 0) {
            throw std::runtime_error("No suitable cuBLASLt algorithm found for FP4 GEMM");
        }
        TLLM_LOG_INFO("[CublasLtFp4GemmRunner::executeCublasLtGemm] Found " + std::to_string(returnedResults) + " suitable algorithms");
        
        // 执行 matmul
        TLLM_LOG_INFO("[CublasLtFp4GemmRunner::executeCublasLtGemm] Executing cuBLASLt matmul");
        float alpha = 1.0f;
        float beta = 0.0f;
        
        // In cuBLASLt FP4 GEMM:
        // - C is the intermediate result (bfloat16) - this is our desired final output type
        // - D is the final output (FP4) - we don't need this for our T-typed output
        TLLM_CUDA_CHECK(cublasLtMatmul(mCublasLtHandle,
                                     operationDesc,
                                     &alpha,
                                     A, Adesc,
                                     B, Bdesc,
                                     &beta,
                                     D, Cdesc,  // Output to D (user's buffer), using Cdesc's type (T)
                                     nullptr, nullptr,  // No D output needed
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceBytes,
                                     0));
        
        TLLM_LOG_INFO("[CublasLtFp4GemmRunner::executeCublasLtGemm] cuBLASLt matmul completed successfully");
        
        // 清理资源
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (Ddesc) cublasLtMatrixLayoutDestroy(Ddesc);
        if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
        if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
        if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
        if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
    }
    catch (...)
    {
        // 清理资源
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (Ddesc) cublasLtMatrixLayoutDestroy(Ddesc);
        if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
        if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
        if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
        if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
        throw;
    }
}

template <typename T>
size_t CublasLtFp4GemmRunner<T>::getWorkspaceSize(int const m, int const n, int const k, int batch_count)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
    // 返回 cuBLASLt 所需的 workspace 大小
    // 这里可以根据矩阵大小估算 workspace 需求
    return std::max(1024 * 1024, m * n * k / 4); // 简单的估算
}

} // namespace cublaslt_kernels
} // namespace kernels
} // namespace tensorrt_llm
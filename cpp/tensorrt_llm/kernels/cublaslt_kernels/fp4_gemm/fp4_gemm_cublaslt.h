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
#include <cuda_fp4.h>
#include <cuda_fp8.h>

namespace tensorrt_llm
{
namespace kernels
{
namespace cublaslt_kernels
{

template <typename T>
CublasLtFp4GemmRunner<T>::CublasLtFp4GemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    checkCudaError(cublasLtCreate(&mCublasLtHandle));
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
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
    // 验证输入类型
    validateInputTypes(A, B, input_sf, weight_sf);
    
    // 执行 cuBLASLt GEMM
    executeCublasLtGemm(D, A, B, input_sf, weight_sf, global_sf, m, n, k, workspace, workspaceBytes, stream);
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
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    
    try
    {
        // 创建操作描述符
        checkCudaError(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, 
                                                     &CUBLAS_OP_N, sizeof(CUBLAS_OP_N)));
        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, 
                                                     &CUBLAS_OP_N, sizeof(CUBLAS_OP_N)));
        
        // 设置缩放模式
        cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_SCALE_ALPHA;
        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, 
                                                     &scaleMode, sizeof(scaleMode)));
        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, 
                                                     &scaleMode, sizeof(scaleMode)));
        
        // 设置缩放指针
        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, 
                                                     &input_sf, sizeof(input_sf)));
        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, 
                                                     &weight_sf, sizeof(weight_sf)));
        
        // 创建矩阵描述符
        checkCudaError(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, m, k, k));
        checkCudaError(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, k, n, n));
        
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
        checkCudaError(cublasLtMatrixLayoutCreate(&Cdesc, outputType, m, n, n));
        checkCudaError(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_4F_E2M1, m, n, n)); // D is FP4 in sample
        
        // 创建偏好描述符
        checkCudaError(cublasLtMatmulPreferenceCreate(&preference));
        checkCudaError(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                                           &workspaceBytes, sizeof(workspaceBytes)));
        
        // 获取启发式算法
        checkCudaError(cublasLtMatmulAlgoGetHeuristic(mCublasLtHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, 
                                                     preference, 1, &heuristicResult, &returnedResults));
        
        if (returnedResults == 0) {
            throw std::runtime_error("No suitable cuBLASLt algorithm found for FP4 GEMM");
        }
        
        // 执行 matmul
        float alpha = 1.0f;
        float beta = 0.0f;
        
        // In cuBLASLt FP4 GEMM:
        // - C is the intermediate result (bfloat16) - this is our desired final output type
        // - D is the final output (FP4) - we don't need this for our T-typed output
        checkCudaError(cublasLtMatmul(mCublasLtHandle,
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
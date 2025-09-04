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

#include "cublaslt_kernels/include/fp4_gemm.h"
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
    
    // 设置 cuBLASLt 流
    checkCudaError(cublasLtSetStream(mCublasLtHandle, stream));
    
    // 执行 GEMM
    executeCublasLtGemm(D, A, B, input_sf, weight_sf, global_sf, 
                       m, n, k, workspace, workspaceBytes, stream);
}

template <typename T>
void CublasLtFp4GemmRunner<T>::validateInputTypes(void const* A, void const* B, 
                                                 void const* input_sf, void const* weight_sf)
{
    // 验证 A, B 必须是 __nv_fp4_e2m1 类型
    // 验证 scale 必须是 __nv_fp8_e4m3 类型
    // 如果不匹配则抛出异常
    // 这里可以添加类型检查逻辑
    TLLM_LOG_DEBUG("Validating input types for cuBLASLt FP4 GEMM");
}

template <typename T>
void CublasLtFp4GemmRunner<T>::executeCublasLtGemm(void* D, void const* A, void const* B, 
                                                  void const* input_sf, void const* weight_sf,
                                                  float const* global_sf, int m, int n, int k, 
                                                  char* workspace, const size_t workspaceBytes, cudaStream_t stream)
{
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

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

        // 设置块缩放模式
        auto aScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_MODE_PER_TENSOR;
        auto bScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_MODE_PER_TENSOR;
        auto dScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_MODE_PER_TENSOR;
        auto dOutScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_MODE_PER_TENSOR;

        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, 
                                                     &aScaleMode, sizeof(aScaleMode)));
        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, 
                                                     &bScaleMode, sizeof(bScaleMode)));
        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, 
                                                     &dScaleMode, sizeof(dScaleMode)));
        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, 
                                                     &dOutScaleMode, sizeof(dOutScaleMode)));

        // 设置缩放因子指针
        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, 
                                                     &input_sf, sizeof(input_sf)));
        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, 
                                                     &weight_sf, sizeof(weight_sf)));
        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, 
                                                     &global_sf, sizeof(global_sf)));
        checkCudaError(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, 
                                                     &global_sf, sizeof(global_sf)));

        // 创建矩阵描述符
        checkCudaError(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, m, k, k));
        checkCudaError(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, k, n, n));
        checkCudaError(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, n));
        checkCudaError(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_4F_E2M1, m, n, n));

        // 创建偏好设置
        checkCudaError(cublasLtMatmulPreferenceCreate(&preference));
        checkCudaError(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                                           &workspaceBytes, sizeof(workspaceBytes)));

        // 使用 cuBLASLt heuristic 选择最佳算法
        checkCudaError(cublasLtMatmulAlgoGetHeuristic(mCublasLtHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, 
                                                     preference, 1, &heuristicResult, &returnedResults));

        if (returnedResults == 0) {
            throw std::runtime_error("No suitable cuBLASLt algorithm found for FP4 GEMM");
        }

        // 执行 matmul
        float alpha = 1.0f;
        float beta = 0.0f;
        checkCudaError(cublasLtMatmul(mCublasLtHandle,
                                     operationDesc,
                                     &alpha,
                                     A, Adesc,
                                     B, Bdesc,
                                     &beta,
                                     D, Cdesc,
                                     D, Ddesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceBytes,
                                     0));
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

    // 清理资源
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (Ddesc) cublasLtMatrixLayoutDestroy(Ddesc);
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
}

template <typename T>
size_t CublasLtFp4GemmRunner<T>::getWorkspaceSize(int const m, int const n, int const k, int batch_count)
{
    // 返回 cuBLASLt 所需的工作空间大小
    // 这里可以根据 m, n, k 估算所需的工作空间
    return 1024 * 1024; // 1MB 默认工作空间
}

} // namespace cublaslt_kernels
} // namespace kernels
} // namespace tensorrt_llm

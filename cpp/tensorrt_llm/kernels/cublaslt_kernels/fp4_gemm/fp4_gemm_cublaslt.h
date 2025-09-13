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

// 基类定义
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
class CublasLtFp4GemmRunner : public CublasLtFp4GemmRunnerInterface
{
public:
    CublasLtFp4GemmRunner();
    ~CublasLtFp4GemmRunner();
    
    void gemm(void* D, void const* A, void const* B, 
             void const* input_sf, void const* weight_sf,
             float const* global_sf, int m, int n, int k, 
             int batch_count, char* workspace, const size_t workspaceBytes, 
             cudaStream_t stream) override;
    
    // 新增：支持不同scaling factor类型的重载
    void gemm(void* D, void const* A, void const* B, 
             void const* input_sf, void const* weight_sf,
             float const* global_sf, int m, int n, int k, 
             int batch_count, char* workspace, const size_t workspaceBytes, 
             cudaStream_t stream, bool input_sf_is_uint8 = true);
    
    size_t getWorkspaceSize(int const m, int const n, int const k, int batch_count) override;

private:
    void validateInputTypes(void const* A, void const* B, 
                           void const* input_sf, void const* weight_sf);
    void executeCublasLtGemm(void* D, void const* A, void const* B, 
                            void const* input_sf, void const* weight_sf,
                            float const* global_sf, int m, int n, int k, 
                            char* workspace, const size_t workspaceBytes, cudaStream_t stream,
                            bool input_sf_is_uint8 = true);
                            // Note: C matrix support can be added later for D = α * A * B + β * C
    
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
    // 默认假设scaling factor是uint8类型（与CUTLASS一致）
    gemm(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, workspace, workspaceBytes, stream, true);
}

template <typename T>
void CublasLtFp4GemmRunner<T>::gemm(void* D, void const* A, void const* B,
                                    void const* input_sf, void const* weight_sf,
                                    float const* global_sf, int m, int n, int k,
                                    int batch_count, char* workspace, const size_t workspaceBytes,
                                    cudaStream_t stream, bool input_sf_is_uint8)
{
    TLLM_LOG_INFO("[CublasLtFp4GemmRunner::gemm] Starting cuBLASLt FP4 GEMM");
    TLLM_LOG_INFO("[CublasLtFp4GemmRunner::gemm] GEMM dimensions: m=" + std::to_string(m) + 
                  ", n=" + std::to_string(n) + ", k=" + std::to_string(k) + 
                  ", batch_count=" + std::to_string(batch_count));
    TLLM_LOG_INFO("[CublasLtFp4GemmRunner::gemm] Workspace size: " + std::to_string(workspaceBytes) + " bytes");
    TLLM_LOG_INFO("[CublasLtFp4GemmRunner::gemm] Output type: " + std::string(typeid(T).name()));
    TLLM_LOG_INFO("[CublasLtFp4GemmRunner::gemm] Input scaling factor type: " + 
                  std::string(input_sf_is_uint8 ? "uint8" : "float8_e4m3fn"));
    
    // 验证输入类型
    validateInputTypes(A, B, input_sf, weight_sf);
    
    // 执行 cuBLASLt GEMM
    executeCublasLtGemm(D, A, B, input_sf, weight_sf, global_sf, m, n, k, workspace, workspaceBytes, stream, input_sf_is_uint8);
    
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
                                                   char* workspace, const size_t workspaceBytes, cudaStream_t stream,
                                                   bool input_sf_is_uint8)
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
        cublasOperation_t transa = CUBLAS_OP_T;
        cublasOperation_t transb = CUBLAS_OP_N;
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, 
                                                     &transa, sizeof(transa)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, 
                                                     &transb, sizeof(transb)));
        
        // 设置缩放模式 - cuBLASLt需要使用e4m3格式的scaling factor
        cublasLtMatmulMatrixScale_t AScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulMatrixScale_t BScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulMatrixScale_t CScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        cublasLtMatmulMatrixScale_t DScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        cublasLtMatmulMatrixScale_t DOutScaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
        
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, 
                                                     &AScaleMode, sizeof(AScaleMode)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, 
                                                     &BScaleMode, sizeof(BScaleMode)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE, 
                                                     &CScaleMode, sizeof(CScaleMode)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, 
                                                     &DScaleMode, sizeof(DScaleMode)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, 
                                                     &DOutScaleMode, sizeof(DOutScaleMode)));
        
        // 设置缩放指针 - cuBLASLt期望e4m3格式的scaling factor
        const void* input_sf_ptr = input_sf;
        const void* weight_sf_ptr = weight_sf;
        
        if (input_sf_is_uint8) {
            // 输入是uint8类型，cuBLASLt期望__nv_fp8_e4m3类型
            // 由于bit pattern相同，可以直接reinterpret_cast为正确的类型
            TLLM_LOG_DEBUG("[CublasLtFp4GemmRunner::executeCublasLtGemm] Converting uint8 scaling factors to __nv_fp8_e4m3");
            input_sf_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(input_sf);
            weight_sf_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(weight_sf);
        } else {
            // 输入已经是__nv_fp8_e4m3类型，直接使用
            TLLM_LOG_DEBUG("[CublasLtFp4GemmRunner::executeCublasLtGemm] Using __nv_fp8_e4m3 scaling factors directly");
            input_sf_ptr = input_sf;
            weight_sf_ptr = weight_sf;
        }
        
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, 
                                                     &input_sf_ptr, sizeof(input_sf_ptr)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, 
                                                     &weight_sf_ptr, sizeof(weight_sf_ptr)));
        
        // 为 C 和 D 矩阵设置缩放指针（根据 cuBLASLt 样本需要）
        // 注意：这里我们使用 nullptr，因为当前实现不需要 C 和 D 矩阵的缩放
        const void* c_scale_ptr = nullptr;
        const void* d_scale_ptr = nullptr;
        const void* d_out_scale_ptr = nullptr;
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, 
                                                     &c_scale_ptr, sizeof(c_scale_ptr)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, 
                                                     &d_scale_ptr, sizeof(d_scale_ptr)));
        TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, 
                                                     &d_out_scale_ptr, sizeof(d_out_scale_ptr)));
        
        // 创建矩阵描述符
        TLLM_LOG_INFO("[CublasLtFp4GemmRunner::executeCublasLtGemm] Creating matrix descriptors");
        // 对于 FP4 矩阵，步长应该是压缩后的维度
        // 注意：Python 端已经交换了输入顺序，所以这里保持原始顺序
        TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, k, m, k));  // A: act_fp4 [k, m]
        TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, k, n, k));  // B: weight [k, n]
        
        
        TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, n));
        TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, m, n, n));
        
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
        
        // In cuBLASLt FP4 GEMM (根据样本):
        // - A, B: FP4 input matrices (CUDA_R_4F_E2M1)
        // - C: output matrix (bfloat16) - 这是用户期望的输出
        // - D: internal FP4 representation (cuBLASLt 内部使用)
        // Current implementation: β = 0, so no C input needed
        // Output to C matrix (bfloat16) which is the user's buffer
        TLLM_CUDA_CHECK(cublasLtMatmul(mCublasLtHandle,
                                     operationDesc,
                                     &alpha,
                                     A, Adesc,  // A: act_fp4 [k, m] - Python 端已交换顺序
                                     B, Bdesc,  // B: weight [k, n] - Python 端已交换顺序
                                     &beta,
                                     nullptr, Cdesc,  // No C input needed (β = 0)
                                     D, Ddesc,  // Output to D buffer using Cdesc (bfloat16) layout
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
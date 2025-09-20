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
#include <mutex>

namespace tensorrt_llm
{
namespace kernels
{
namespace cublaslt_kernels
{

// Device-side constant zero value
__device__ __constant__ float zero_device;

// Function to get device pointer to constant zero
inline float* GetScalarZero() {
    static std::once_flag init_flag;
    std::call_once(init_flag, []() {
        float zero = 0.0f;
        TLLM_CUDA_CHECK(cudaMemcpyToSymbol(&zero_device, &zero, sizeof(float)));
    });
    // return address by cudaGetSymbolAddress
    float* dev_ptr;
    TLLM_CUDA_CHECK(cudaGetSymbolAddress((void**)&dev_ptr, &zero_device));
    return dev_ptr;
}

// Base class definition
class CublasLtFp4GemmRunnerInterface
{
public:
    virtual ~CublasLtFp4GemmRunnerInterface() = default;
    
    virtual void gemm(void* D, void const* A, void const* B, 
                     void const* a_sf, void const* b_sf,
                     float const* global_sf, int m, int n, int k, 
                     int batch_count, char* workspace, const size_t workspaceBytes, 
                     cudaStream_t stream) = 0;
    
    virtual size_t getWorkspaceSize() = 0;
};

// Template class definition
template <typename T>
class CublasLtFp4GemmRunner : public CublasLtFp4GemmRunnerInterface
{
public:
    CublasLtFp4GemmRunner();
    ~CublasLtFp4GemmRunner();
    
    void gemm(void* D, void const* A, void const* B, 
             void const* a_sf, void const* b_sf,
             float const* global_sf, int m, int n, int k, 
             int batch_count, char* workspace, const size_t workspaceBytes, 
             cudaStream_t stream) override;
    
    size_t getWorkspaceSize() override;

private:
    void executeCublasLtGemm(void* D, void const* A, void const* B,
                            void const* a_sf, void const* b_sf,
                            float const* global_sf, int m, int n, int k, 
                            char* workspace, const size_t workspaceBytes, cudaStream_t stream);
                            // Note: C matrix support can be added later for D = α * A * B + β * C
    
    cublasLtHandle_t mCublasLtHandle;
};

// Template class implementation
template <typename T>
CublasLtFp4GemmRunner<T>::CublasLtFp4GemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    TLLM_CUDA_CHECK(cublasLtCreate(&mCublasLtHandle));
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
                                    void const* a_sf, void const* b_sf,
                                    float const* global_sf, int m, int n, int k,
                                    int batch_count, char* workspace, const size_t workspaceBytes,
                                    cudaStream_t stream)
{
    if (batch_count > 1)
    {
        throw std::runtime_error("[TensorRT-LLM Error][FP4] CublasLtFp4GemmRunner: batch_count > 1 is not supported yet.");
        
    }
    // Execute cuBLASLt GEMM
    executeCublasLtGemm(D, A, B, a_sf, b_sf, global_sf, m, n, k, workspace, workspaceBytes, stream);
    
}


template <typename T>
void CublasLtFp4GemmRunner<T>::executeCublasLtGemm(void* D, void const* A, void const* B,
                                                   void const* a_sf, void const* b_sf,
                                                   float const* global_sf, int m, int n, int k,
                                                   char* workspace, const size_t workspaceBytes, cudaStream_t stream)
{
    
    // Support fp16, bf16, and fp32 output types
    cudaDataType_t output_dtype;

    if (std::is_same<T, half>::value) {
        output_dtype = CUDA_R_16F;
    } else if (std::is_same<T, __nv_bfloat16>::value) {
        output_dtype = CUDA_R_16BF;
    } else if (std::is_same<T, float>::value) {
        output_dtype = CUDA_R_32F;
    } else {
        throw std::runtime_error("CublasLtFp4GemmRunner: Unsupported output type");
    }

    
    cublasLtMatmulDesc_t operationDesc = nullptr;
    cublasLtMatrixLayout_t Adesc = nullptr, Bdesc = nullptr, Cdesc = nullptr, Ddesc = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;
    
    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    
    // Create operation descriptor
    TLLM_CUDA_CHECK(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, 
                                                 &transa, sizeof(transa)));
    TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, 
                                                 &transb, sizeof(transb)));

    cublasLtPointerMode_t pointer_mode = CUBLASLT_POINTER_MODE_DEVICE;
    TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_POINTER_MODE, &pointer_mode, sizeof(pointer_mode)));
    
    // Set scaling mode - cuBLASLt requires e4m3 format scaling factors
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
    
    const void* a_sf_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(a_sf);
    const void* b_sf_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(b_sf);
        
    TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, 
                                                      &a_sf_ptr, sizeof(a_sf_ptr)));
    TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, 
                                                 &b_sf_ptr, sizeof(b_sf_ptr)));
    
    const void* c_scale_ptr = nullptr;
    const void* d_scale_ptr = nullptr;
    const void* d_out_scale_ptr = nullptr;
    TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_POINTER, 
                                                 &c_scale_ptr, sizeof(c_scale_ptr)));
    TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, 
                                                 &d_scale_ptr, sizeof(d_scale_ptr)));
    TLLM_CUDA_CHECK(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, 
                                                 &d_out_scale_ptr, sizeof(d_out_scale_ptr)));
    
    TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, k, m, k));
    TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, k, n, k));
    
    
    TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Cdesc, output_dtype, m, n, m));
    TLLM_CUDA_CHECK(cublasLtMatrixLayoutCreate(&Ddesc, output_dtype, m, n, m));
    
    // Create preference descriptor
    TLLM_CUDA_CHECK(cublasLtMatmulPreferenceCreate(&preference));
    TLLM_CUDA_CHECK(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 
                                                       &workspaceBytes, sizeof(workspaceBytes)));
    
    // Get heuristic algorithm
    TLLM_CUDA_CHECK(cublasLtMatmulAlgoGetHeuristic(mCublasLtHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, 
                                                 preference, 1, &heuristicResult, &returnedResults));
    
    if (returnedResults == 0) {
        throw std::runtime_error("No suitable cuBLASLt algorithm found for FP4 GEMM");
    }
    
    
    // Get device pointer to constant zero
    float* d_zero_ptr = GetScalarZero();
    
    TLLM_CUDA_CHECK(cublasLtMatmul(mCublasLtHandle,
                                 operationDesc,
                                 global_sf,
                                 A, Adesc,
                                 B, Bdesc,
                                 d_zero_ptr,
                                 D, Cdesc,
                                 D, Ddesc,
                                 &heuristicResult.algo,
                                 workspace,
                                 workspaceBytes,
                                 stream));
    
    
    // Clean up resources
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (Ddesc) cublasLtMatrixLayoutDestroy(Ddesc);
    if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
    if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
    if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
    if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
}

template <typename T>
size_t CublasLtFp4GemmRunner<T>::getWorkspaceSize()
{
    // 32MB
    return 33554432;
}

} // namespace cublaslt_kernels
} // namespace kernels
} // namespace tensorrt_llm
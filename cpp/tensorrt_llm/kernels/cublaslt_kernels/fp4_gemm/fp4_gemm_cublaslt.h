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

// Function to get device pointer to constant zero
float* GetScalarZero();

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

// Template function declarations - implementations moved to .cpp file
template <typename T>
void CublasLtFp4GemmRunner<T>::gemm(void* D, void const* A, void const* B,
                                    void const* a_sf, void const* b_sf,
                                    float const* global_sf, int m, int n, int k,
                                    int batch_count, char* workspace, const size_t workspaceBytes,
                                    cudaStream_t stream);

template <typename T>
void CublasLtFp4GemmRunner<T>::executeCublasLtGemm(void* D, void const* A, void const* B,
                                                   void const* a_sf, void const* b_sf,
                                                   float const* global_sf, int m, int n, int k,
                                                   char* workspace, const size_t workspaceBytes, cudaStream_t stream);

template <typename T>
size_t CublasLtFp4GemmRunner<T>::getWorkspaceSize();

} // namespace cublaslt_kernels
} // namespace kernels
} // namespace tensorrt_llm
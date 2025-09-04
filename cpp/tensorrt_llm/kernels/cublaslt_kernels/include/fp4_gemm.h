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

#include <cuda_runtime_api.h>
#include <cublasLt.h>
#include <vector>

namespace tensorrt_llm
{
namespace kernels
{
namespace cublaslt_kernels
{

// cuBLASLt FP4 GEMM Runner 接口
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

// 模板化实现
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

} // namespace cublaslt_kernels
} // namespace kernels
} // namespace tensorrt_llm

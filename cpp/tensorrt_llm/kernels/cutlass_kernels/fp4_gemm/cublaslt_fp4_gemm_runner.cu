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

#include "../include/fp4_gemm.h"
#include "tensorrt_llm/common/logger.h"
#include "cublaslt_nvfp4_gemm.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cublaslt_kernels
{

template <typename T, fp4_gemm::FP4GemmType gemmType>
CublasLtFp4GemmRunner<T, gemmType>::CublasLtFp4GemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
#ifdef ENABLE_CUBLASLT_FP4
    TLLM_LOG_INFO("Initializing cuBLASLt FP4 GEMM Runner");
#else
    throw std::runtime_error("cuBLASLt FP4 GEMM backend not enabled in this build");
#endif
}

template <typename T, fp4_gemm::FP4GemmType gemmType>
CublasLtFp4GemmRunner<T, gemmType>::~CublasLtFp4GemmRunner()
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
}

template <typename T, fp4_gemm::FP4GemmType gemmType>
void CublasLtFp4GemmRunner<T, gemmType>::gemm(void* D, void const* A, void const* B, void const* input_sf, 
    void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
    tkc::CutlassGemmConfig gemmConfig, char* workspace, const size_t workspaceBytes, cudaStream_t stream)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
#ifdef ENABLE_CUBLASLT_FP4
    // 检查是否支持当前的 GEMM 类型
    if (!supportsGemmType(gemmType)) {
        throw std::runtime_error("cuBLASLt backend does not support this GEMM type");
    }
    
    executeCublasLtGemm(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, 
                       workspace, workspaceBytes, stream);
#else
    throw std::runtime_error("cuBLASLt FP4 GEMM backend not enabled in this build");
#endif
}

template <typename T, fp4_gemm::FP4GemmType gemmType>
size_t CublasLtFp4GemmRunner<T, gemmType>::getWorkspaceSize(int const m, int const n, int const k, int batch_count)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
#ifdef ENABLE_CUBLASLT_FP4
    return getCublasLtNvfp4GemmWorkspaceSize(m, n, k, batch_count);
#else
    return 0;
#endif
}

template <typename T, fp4_gemm::FP4GemmType gemmType>
std::vector<tkc::CutlassGemmConfig> CublasLtFp4GemmRunner<T, gemmType>::getConfigs() const
{
    // cuBLASLt 使用启发式算法，不需要预定义的配置
    return {};
}

template <typename T, fp4_gemm::FP4GemmType gemmType>
bool CublasLtFp4GemmRunner<T, gemmType>::supportsGemmType(fp4_gemm::FP4GemmType type) const
{
    // cuBLASLt 目前仅支持 W4A4_NVFP4_NVFP4
    return gemmType == fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4;
}

template <typename T, fp4_gemm::FP4GemmType gemmType>
void CublasLtFp4GemmRunner<T, gemmType>::executeCublasLtGemm(void* D, void const* A, void const* B, 
    void const* input_sf, void const* weight_sf, float const* global_sf, int m, int n, int k, int batch_count,
    char* workspace, const size_t workspaceBytes, cudaStream_t stream)
{
#ifdef ENABLE_CUBLASLT_FP4
    TLLM_LOG_DEBUG("Executing cuBLASLt FP4 GEMM");
    
    // 调用 cuBLASLt 实现
    if constexpr (std::is_same_v<T, half>)
    {
        executeCublasLtNvfp4GemmFp16(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, 
            workspace, workspaceBytes, stream);
    }
    else if constexpr (std::is_same_v<T, __nv_bfloat16>)
    {
#ifdef ENABLE_BF16
        executeCublasLtNvfp4GemmBf16(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, 
            workspace, workspaceBytes, stream);
#else
        throw std::runtime_error("BF16 not enabled in this build");
#endif
    }
    else if constexpr (std::is_same_v<T, float>)
    {
        executeCublasLtNvfp4GemmFp32(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, 
            workspace, workspaceBytes, stream);
    }
    else
    {
        throw std::runtime_error("Unsupported output type for cuBLASLt NVFP4 GEMM");
    }
#else
    throw std::runtime_error("cuBLASLt FP4 GEMM backend not enabled in this build");
#endif
}

// 显式实例化
#ifdef ENABLE_CUBLASLT_FP4
template class CublasLtFp4GemmRunner<half, fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4>;
#ifdef ENABLE_BF16
template class CublasLtFp4GemmRunner<__nv_bfloat16, fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4>;
#endif
template class CublasLtFp4GemmRunner<float, fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4>;
#endif

} // namespace cublaslt_kernels
} // namespace kernels
} // namespace tensorrt_llm

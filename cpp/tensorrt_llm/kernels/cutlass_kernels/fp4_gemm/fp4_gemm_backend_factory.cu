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

namespace tensorrt_llm
{
namespace kernels
{
namespace fp4_gemm
{

template <typename T>
std::unique_ptr<IFp4GemmRunner> Fp4GemmBackendFactory::createRunner(
    FP4GemmType type, 
    FP4GemmBackend backend)
{
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    
    switch (backend)
    {
    case FP4GemmBackend::CUTLASS:
        TLLM_LOG_DEBUG("Creating CUTLASS FP4 GEMM Runner");
        if (type == FP4GemmType::W4A4_NVFP4_NVFP4) {
            return std::make_unique<cutlass_kernels::CutlassFp4GemmRunner<T, FP4GemmType::W4A4_NVFP4_NVFP4>>();
        } else if (type == FP4GemmType::W4A8_MXFP4_MXFP8) {
            return std::make_unique<cutlass_kernels::CutlassFp4GemmRunner<T, FP4GemmType::W4A8_MXFP4_MXFP8>>();
        } else {
            throw std::runtime_error("Unsupported GEMM type for CUTLASS backend");
        }
        break;
        
    case FP4GemmBackend::CUBLASLT:
        TLLM_LOG_DEBUG("Creating cuBLASLt FP4 GEMM Runner");
#ifdef ENABLE_CUBLASLT_FP4
        if (type == FP4GemmType::W4A4_NVFP4_NVFP4) {
            return std::make_unique<cublaslt_kernels::CublasLtFp4GemmRunner<T, FP4GemmType::W4A4_NVFP4_NVFP4>>();
        } else {
            throw std::runtime_error("cuBLASLt backend only supports W4A4_NVFP4_NVFP4 GEMM type");
        }
#else
        throw std::runtime_error("cuBLASLt backend not enabled in this build");
#endif
        break;
        
    default:
        throw std::runtime_error("Unknown FP4 GEMM backend");
    }
}

std::vector<FP4GemmBackend> Fp4GemmBackendFactory::getAvailableBackends()
{
    std::vector<FP4GemmBackend> backends;
    
    // CUTLASS 后端总是可用的
    backends.push_back(FP4GemmBackend::CUTLASS);
    
    // cuBLASLt 后端仅在编译时启用时可用
#ifdef ENABLE_CUBLASLT_FP4
    backends.push_back(FP4GemmBackend::CUBLASLT);
#endif
    
    return backends;
}

bool Fp4GemmBackendFactory::isBackendAvailable(FP4GemmBackend backend)
{
    switch (backend)
    {
    case FP4GemmBackend::CUTLASS:
        return true;
    case FP4GemmBackend::CUBLASLT:
#ifdef ENABLE_CUBLASLT_FP4
        return true;
#else
        return false;
#endif
    default:
        return false;
    }
}

FP4GemmBackend Fp4GemmBackendFactory::getRecommendedBackend(FP4GemmType type)
{
    switch (type)
    {
    case FP4GemmType::W4A4_NVFP4_NVFP4:
        // 对于 W4A4，优先使用 cuBLASLt（如果可用）
#ifdef ENABLE_CUBLASLT_FP4
        return FP4GemmBackend::CUBLASLT;
#else
        return FP4GemmBackend::CUTLASS;
#endif
    case FP4GemmType::W4A8_MXFP4_MXFP8:
        // 对于 W4A8，只能使用 CUTLASS
        return FP4GemmBackend::CUTLASS;
    default:
        return FP4GemmBackend::CUTLASS;
    }
}

// 显式实例化
template std::unique_ptr<IFp4GemmRunner> Fp4GemmBackendFactory::createRunner<half>(FP4GemmType, FP4GemmBackend);
#ifdef ENABLE_BF16
template std::unique_ptr<IFp4GemmRunner> Fp4GemmBackendFactory::createRunner<__nv_bfloat16>(FP4GemmType, FP4GemmBackend);
#endif
template std::unique_ptr<IFp4GemmRunner> Fp4GemmBackendFactory::createRunner<float>(FP4GemmType, FP4GemmBackend);

} // namespace fp4_gemm
} // namespace kernels
} // namespace tensorrt_llm

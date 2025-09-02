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
#include <vector>

#include "cutlass_extensions/gemm_configs.h"
#include "tensorrt_llm/common/quantization.h"

namespace tk = tensorrt_llm::common;
namespace tkc = tensorrt_llm::cutlass_extensions;

namespace tensorrt_llm
{
namespace kernels
{
namespace fp4_gemm
{

/*
  This runner supports:
  FP4 inputs (A and B)
  float blockwise scaling factor
  float alpha scalings
  T output (D) where T = {float, half, __nv_bfloat16}

  Activations, biases and outputs are all assumed to be row-major.
  Weights are assumed to be column-major.
  Block scaling factor are interleaved.
*/

// FP4 GEMM 类型枚举
enum class FP4GemmType
{
    W4A4_NVFP4_NVFP4,
    W4A8_MXFP4_MXFP8,
};

// FP4 GEMM 后端类型枚举
enum class FP4GemmBackend
{
    CUTLASS,
    CUBLASLT,
};

// 统一的 FP4 GEMM 接口
class IFp4GemmRunner
{
public:
    virtual ~IFp4GemmRunner() = default;

    // 执行 FP4 GEMM 操作
    virtual void gemm(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
        float const* global_sf, int m, int n, int k, int batch_count, tkc::CutlassGemmConfig gemmConfig,
        char* workspace, const size_t workspaceBytes, cudaStream_t stream) = 0;

    // 获取工作空间大小
    virtual size_t getWorkspaceSize(int const m, int const n, int const k, int batch_count) = 0;

    // 获取支持的配置
    virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;

    // 获取后端类型
    virtual FP4GemmBackend getBackendType() const = 0;

    // 检查是否支持指定的 GEMM 类型
    virtual bool supportsGemmType(FP4GemmType gemmType) const = 0;
};

// 后端工厂类
class Fp4GemmBackendFactory
{
public:
    // 创建 FP4 GEMM Runner
    template <typename T>
    static std::unique_ptr<IFp4GemmRunner> createRunner(
        FP4GemmType gemmType, 
        FP4GemmBackend backend = FP4GemmBackend::CUTLASS);

    // 获取可用的后端列表
    static std::vector<FP4GemmBackend> getAvailableBackends();

    // 检查后端是否可用
    static bool isBackendAvailable(FP4GemmBackend backend);

    // 获取推荐的后端
    static FP4GemmBackend getRecommendedBackend(FP4GemmType gemmType);
};

// 为了向后兼容，保留原来的接口名称
using CutlassFp4GemmRunnerInterface = IFp4GemmRunner;

} // namespace fp4_gemm

// CUTLASS 相关的类放在 cutlass_kernels 命名空间
namespace cutlass_kernels
{

template <typename T, fp4_gemm::FP4GemmType gemmType = fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4>
class CutlassFp4GemmRunner : public virtual fp4_gemm::IFp4GemmRunner
{
public:
    CutlassFp4GemmRunner();
    ~CutlassFp4GemmRunner();

    void gemm(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
        float const* global_sf, int m, int n, int k, int batch_count, tkc::CutlassGemmConfig gemmConfig,
        char* workspace, const size_t workspaceBytes, cudaStream_t stream) override;

    // Returns desired workspace size in bytes.
    size_t getWorkspaceSize(int const m, int const n, int const k, int const batch_count) override;

    std::vector<tkc::CutlassGemmConfig> getConfigs() const override;

    // 实现新的接口方法
    fp4_gemm::FP4GemmBackend getBackendType() const override { return fp4_gemm::FP4GemmBackend::CUTLASS; }
    bool supportsGemmType(fp4_gemm::FP4GemmType type) const override;

private:
    size_t dispatchToArch(T* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
        float const* global_sf, int m, int n, int k, int batch_count, tkc::CutlassGemmConfig gemmConfig,
        char* workspace, const size_t workspaceBytes, cudaStream_t stream, int* occupancy = nullptr);

    size_t getWorkspaceSizeImpl(int const m, int const n, int const k, int const batch_count);

    int mSm;
    int mMultiProcessorCount;
};

} // namespace cutlass_kernels

// cuBLASLt 相关的类放在 cublaslt_kernels 命名空间
namespace cublaslt_kernels
{

// cuBLASLt FP4 GEMM Runner 实现
template <typename T, fp4_gemm::FP4GemmType gemmType>
class CublasLtFp4GemmRunner : public fp4_gemm::IFp4GemmRunner
{
public:
    CublasLtFp4GemmRunner();
    ~CublasLtFp4GemmRunner() override;

    // IFp4GemmRunner 接口实现
    void gemm(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
        float const* global_sf, int m, int n, int k, int batch_count, tkc::CutlassGemmConfig gemmConfig,
        char* workspace, const size_t workspaceBytes, cudaStream_t stream) override;

    size_t getWorkspaceSize(int const m, int const n, int const k, int batch_count) override;
    std::vector<tkc::CutlassGemmConfig> getConfigs() const override;
    fp4_gemm::FP4GemmBackend getBackendType() const override { return fp4_gemm::FP4GemmBackend::CUBLASLT; }
    bool supportsGemmType(fp4_gemm::FP4GemmType type) const override;

private:
    // cuBLASLt 特定的实现
    void executeCublasLtGemm(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
        float const* global_sf, int m, int n, int k, int batch_count, char* workspace,
        const size_t workspaceBytes, cudaStream_t stream);
};

} // namespace cublaslt_kernels
} // namespace kernels
} // namespace tensorrt_llm

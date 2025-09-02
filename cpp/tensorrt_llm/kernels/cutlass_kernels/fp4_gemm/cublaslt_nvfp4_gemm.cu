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

#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace cublaslt_kernels
{

using namespace tensorrt_llm::common;

// cuBLASLt NVFP4 GEMM 包装器类
class CublasLtNvfp4GemmWrapper
{
public:
    CublasLtNvfp4GemmWrapper()
    {
        check_cuda_error(cublasLtCreate(&mHandle));
    }

    ~CublasLtNvfp4GemmWrapper()
    {
        if (mHandle)
        {
            cublasLtDestroy(mHandle);
        }
    }

    // 执行 NVFP4 GEMM 操作
    template <typename T>
    void gemm(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
        float const* global_sf, int m, int n, int k, int batch_count, char* workspace,
        const size_t workspaceBytes, cudaStream_t stream)
    {
        TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);

        // 设置转置操作
        cublasOperation_t transa = CUBLAS_OP_N;  // A 矩阵不转置
        cublasOperation_t transb = CUBLAS_OP_N;  // B 矩阵不转置

        // 调用 cuBLASLt NVFP4 GEMM
        executeNvfp4Gemm<T>(transa, transb, m, n, k, global_sf, 
            static_cast<const __nv_fp8_e4m3*>(input_sf),
            static_cast<const __nv_fp4_e2m1*>(A), k, 
            static_cast<const __nv_fp8_e4m3*>(weight_sf),
            static_cast<const __nv_fp4_e2m1*>(B), k, nullptr,
            nullptr, static_cast<T*>(D), n, nullptr, nullptr, n, nullptr,
            workspace, workspaceBytes, stream);
    }

    // 获取工作空间大小
    size_t getWorkspaceSize(int m, int n, int k, int batch_count)
    {
        // cuBLASLt 通常需要的工作空间大小
        // 这里可以根据实际需求调整
        return 32 * 1024 * 1024; // 32MB
    }

private:
    cublasLtHandle_t mHandle;

    // 执行 NVFP4 GEMM 的核心函数
    template <typename T>
    void executeNvfp4Gemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k,
        const float* alpha, const __nv_fp8_e4m3* a_scale, const __nv_fp4_e2m1* A, int lda,
        const __nv_fp8_e4m3* b_scale, const __nv_fp4_e2m1* B, int ldb, const float* beta,
        const __nv_fp8_e4m3* c_scale, T* C, int ldc, const float* d_scale,
        __nv_fp4_e2m1* D, int ldd, __nv_fp8_e4m3* d_out_scale, void* workspace,
        size_t workspaceSize, cudaStream_t stream)
    {
        cublasLtMatmulDesc_t operationDesc = NULL;
        cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
        cublasLtMatmulPreference_t preference = NULL;

        int returnedResults = 0;
        cublasLtMatmulHeuristicResult_t heuristicResult = {};

        try
        {
            // 创建操作描述符
            checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

            // 设置块缩放模式 - 使用数值常量 (CUBLASLT_MATMUL_MATRIX_SCALE_ALPHA = 0)
            cublasLtMatmulMatrixScale_t AScaleMode = static_cast<cublasLtMatmulMatrixScale_t>(0);
            cublasLtMatmulMatrixScale_t BScaleMode = static_cast<cublasLtMatmulMatrixScale_t>(0);
            cublasLtMatmulMatrixScale_t CScaleMode = static_cast<cublasLtMatmulMatrixScale_t>(0);
            cublasLtMatmulMatrixScale_t DScaleMode = static_cast<cublasLtMatmulMatrixScale_t>(0);
            cublasLtMatmulMatrixScale_t DOutScaleMode = static_cast<cublasLtMatmulMatrixScale_t>(0);

            checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &AScaleMode, sizeof(AScaleMode)));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &BScaleMode, sizeof(BScaleMode)));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_C_SCALE_MODE, &CScaleMode, sizeof(CScaleMode)));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_MODE, &DScaleMode, sizeof(DScaleMode)));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_MODE, &DOutScaleMode, sizeof(DOutScaleMode)));

            // 设置缩放因子指针
            checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, &d_scale, sizeof(d_scale)));
            checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_D_OUT_SCALE_POINTER, &d_out_scale, sizeof(d_out_scale)));

            // 创建矩阵描述符
            checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
            checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
            
            // 根据输出类型创建 C 和 D 描述符
            cudaDataType_t outputType = getCudaDataType<T>();
            checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, outputType, m, n, ldc));
            checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_4F_E2M1, m, n, ldd));

            // 创建偏好句柄
            checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
            checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

            // 获取启发式算法
            checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(mHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

            if (returnedResults == 0)
            {
                throw std::runtime_error("No valid heuristic found for NVFP4 GEMM");
            }

            // 执行矩阵乘法
            checkCublasStatus(cublasLtMatmul(mHandle, operationDesc, alpha, A, Adesc, B, Bdesc, beta, C, Cdesc, D, Ddesc,
                &heuristicResult.algo, workspace, workspaceSize, stream));

            // 同步流
            check_cuda_error(cudaStreamSynchronize(stream));
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
        if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
        if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
        if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
        if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
        if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
        if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    }

    // 获取 CUDA 数据类型
    template <typename T>
    cudaDataType_t getCudaDataType()
    {
        if constexpr (std::is_same_v<T, half>)
        {
            return CUDA_R_16F;
        }
        else if constexpr (std::is_same_v<T, __nv_bfloat16>)
        {
            return CUDA_R_16BF;
        }
        else if constexpr (std::is_same_v<T, float>)
        {
            return CUDA_R_32F;
        }
        else
        {
            static_assert(always_false<T>, "Unsupported output type for cuBLASLt NVFP4 GEMM");
        }
    }

    // 检查 cuBLAS 状态
    void checkCublasStatus(cublasStatus_t status)
    {
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error("cuBLAS error: " + std::to_string(status));
        }
    }

    template <typename T>
    static constexpr bool always_false = false;
};

// 全局 cuBLASLt 包装器实例
static CublasLtNvfp4GemmWrapper gCublasLtWrapper;

// 导出函数：执行 NVFP4 GEMM
extern "C" {

// 执行 NVFP4 GEMM (FP16 输出)
void executeCublasLtNvfp4GemmFp16(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
    float const* global_sf, int m, int n, int k, int batch_count, char* workspace,
    const size_t workspaceBytes, cudaStream_t stream)
{
    gCublasLtWrapper.gemm<half>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, workspace, workspaceBytes, stream);
}

// 执行 NVFP4 GEMM (BF16 输出)
void executeCublasLtNvfp4GemmBf16(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
    float const* global_sf, int m, int n, int k, int batch_count, char* workspace,
    const size_t workspaceBytes, cudaStream_t stream)
{
#ifdef ENABLE_BF16
    gCublasLtWrapper.gemm<__nv_bfloat16>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, workspace, workspaceBytes, stream);
#else
    throw std::runtime_error("BF16 not enabled in this build");
#endif
}

// 执行 NVFP4 GEMM (FP32 输出)
void executeCublasLtNvfp4GemmFp32(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
    float const* global_sf, int m, int n, int k, int batch_count, char* workspace,
    const size_t workspaceBytes, cudaStream_t stream)
{
    gCublasLtWrapper.gemm<float>(D, A, B, input_sf, weight_sf, global_sf, m, n, k, batch_count, workspace, workspaceBytes, stream);
}

// 获取工作空间大小
size_t getCublasLtNvfp4GemmWorkspaceSize(int m, int n, int k, int batch_count)
{
    return gCublasLtWrapper.getWorkspaceSize(m, n, k, batch_count);
}

} // extern "C"

} // namespace cublaslt_kernels
} // namespace kernels
} // namespace tensorrt_llm

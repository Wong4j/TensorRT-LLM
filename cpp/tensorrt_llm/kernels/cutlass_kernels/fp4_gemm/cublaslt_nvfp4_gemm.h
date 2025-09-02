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

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#ifdef __cplusplus
extern "C" {
#endif

// 执行 NVFP4 GEMM (FP16 输出)
void executeCublasLtNvfp4GemmFp16(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
    float const* global_sf, int m, int n, int k, int batch_count, char* workspace,
    const size_t workspaceBytes, cudaStream_t stream);

// 执行 NVFP4 GEMM (BF16 输出)
void executeCublasLtNvfp4GemmBf16(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
    float const* global_sf, int m, int n, int k, int batch_count, char* workspace,
    const size_t workspaceBytes, cudaStream_t stream);

// 执行 NVFP4 GEMM (FP32 输出)
void executeCublasLtNvfp4GemmFp32(void* D, void const* A, void const* B, void const* input_sf, void const* weight_sf,
    float const* global_sf, int m, int n, int k, int batch_count, char* workspace,
    const size_t workspaceBytes, cudaStream_t stream);

// 获取工作空间大小
size_t getCublasLtNvfp4GemmWorkspaceSize(int m, int n, int k, int batch_count);

#ifdef __cplusplus
}
#endif

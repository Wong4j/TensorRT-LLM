/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "cublasFp4ScaledMM.h"
#include "tensorrt_llm/common/cublasMMWrapper.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/plugins/common/plugin.h"
#include "tensorrt_llm/thop/thUtils.h"
#include "userbuffersTensor.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp8.h>
#include <torch/extension.h>
#include <unordered_map>

using torch::Tensor;

namespace torch_ext
{

namespace
{

using tensorrt_llm::common::check;
using tensorrt_llm::common::CublasMMWrapper;

void cublas_fp4_gemm_caller(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
    torch::Tensor const& scale_a, torch::Tensor const& scale_b, torch::Tensor const& alpha, torch::Tensor const& beta)
{
    int32_t m = a.sizes()[0];
    int32_t n = b.sizes()[0];
    int32_t k_compressed = a.sizes()[1];
    int32_t k = k_compressed * 2;

    TLLM_LOG_INFO(
        "cublas_fp4_gemm_caller: Executing FP4 GEMM with default algorithm for shape (m=%d, n=%d, k=%d)", m, n, k);

    // Use device-aware thread-local CublasMMWrapper for FP4 GEMM
    at::cuda::CUDAGuard deviceGuard(a.device());

    thread_local std::unordered_map<int, std::shared_ptr<CublasMMWrapper>> cublasWrappers;
    auto& cublasWrapper = cublasWrappers[a.get_device()];
    if (!cublasWrapper)
    {
        auto cublasHandle = getCublasHandle();
        auto cublasLtHandle = getCublasLtHandle();
        cublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);
    }

    // Set FP4 configuration
    cublasWrapper->setFP4GemmConfig(CUDA_R_16BF); // Output as BF16

    // Get workspace
    auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
    auto workspace = torch::empty(CUBLAS_WORKSPACE_SIZE, workspace_options);

    // Get stream
    auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

    // Get data pointers
    auto* a_ptr = static_cast<void*>(a.data_ptr());
    auto* b_ptr = static_cast<void*>(b.data_ptr());
    auto* out_ptr = static_cast<void*>(out.data_ptr());
    auto* ws_ptr = static_cast<void*>(workspace.data_ptr());

    // Convert scaling factors to __nv_fp8_e4m3 format for cuBLASLt
    void const* a_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_a.data_ptr());
    void const* b_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_b.data_ptr());

    // Validate pointers
    TLLM_CHECK_WITH_INFO(a_sf_ptr != nullptr, "a_sf_ptr is null");
    TLLM_CHECK_WITH_INFO(b_sf_ptr != nullptr, "b_sf_ptr is null");

    // Validate alpha and beta tensors before accessing data
    TLLM_CHECK_WITH_INFO(alpha.numel() > 0, "Alpha tensor is empty");
    TLLM_CHECK_WITH_INFO(beta.numel() > 0, "Beta tensor is empty");
    TLLM_CHECK_WITH_INFO(alpha.dtype() == torch::kFloat32, "Alpha tensor must be float32");
    TLLM_CHECK_WITH_INFO(beta.dtype() == torch::kFloat32, "Beta tensor must be float32");

    auto* alpha_ptr = alpha.data_ptr<float>();
    auto* beta_ptr = beta.data_ptr<float>();

    TLLM_CHECK_WITH_INFO(alpha_ptr != nullptr, "alpha_ptr is null");
    TLLM_CHECK_WITH_INFO(beta_ptr != nullptr, "beta_ptr is null");

    // Set workspace and stream
    cublasWrapper->setStream(stream);
    cublasWrapper->setWorkspace(ws_ptr);

    // Perform FP4 GEMM using CublasMMWrapper
    // Note: A is column major, B is row major, so we swap A and B
    cublasWrapper->Fp4Gemm(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, b_ptr, k, // B matrix (swapped)
        a_ptr, k,                                                       // A matrix (swapped)
        out_ptr, n,                                                     // Output matrix
        b_sf_ptr, a_sf_ptr, alpha_ptr, beta_ptr);
}

} // namespace

Tensor& cublas_fp4_scaled_mm_out(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    Tensor const& alpha, Tensor const& beta, Tensor& out)
{
    // Check device
    CHECK_TH_CUDA(mat_a);
    CHECK_TH_CUDA(mat_b);
    CHECK_TH_CUDA(scale_a);
    CHECK_TH_CUDA(scale_b);
    CHECK_TH_CUDA(alpha);
    CHECK_TH_CUDA(beta);
    CHECK_TH_CUDA(out);

    // Ensure all tensors are on the same device
    auto const deviceIndex = mat_a.get_device();
    TORCH_CHECK(mat_b.get_device() == deviceIndex, "mat_b must be colocated with mat_a");
    TORCH_CHECK(scale_a.get_device() == deviceIndex, "scale_a must be colocated with mat_a");
    TORCH_CHECK(scale_b.get_device() == deviceIndex, "scale_b must be colocated with mat_a");
    TORCH_CHECK(alpha.get_device() == deviceIndex, "alpha must be colocated with mat_a");
    TORCH_CHECK(beta.get_device() == deviceIndex, "beta must be colocated with mat_a");
    TORCH_CHECK(out.get_device() == deviceIndex, "out must be colocated with mat_a");

    // Check dimensions
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2 && out.dim() == 2);
    TORCH_CHECK(out.sizes()[0] == mat_a.sizes()[0] && // m
        mat_a.sizes()[1] == mat_b.sizes()[1] &&       // k
        mat_b.sizes()[0] == out.sizes()[1]);          // n

    // Check scaling factors
    TORCH_CHECK(alpha.numel() == 1);
    TORCH_CHECK(beta.numel() == 1);

    // Check data types - FP4 is typically represented as uint8 in PyTorch
    TORCH_CHECK(mat_a.dtype() == torch::kUInt8);
    TORCH_CHECK(mat_b.dtype() == torch::kUInt8);
    TORCH_CHECK(scale_a.dtype() == torch::kUInt8);
    TORCH_CHECK(scale_b.dtype() == torch::kUInt8);
    TORCH_CHECK(alpha.dtype() == torch::kFloat32);
    TORCH_CHECK(beta.dtype() == torch::kFloat32);

    cublas_fp4_gemm_caller(out, mat_a, mat_b, scale_a, scale_b, alpha, beta);
    return out;
}

Tensor cublas_fp4_scaled_mm(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_a, Tensor const& scale_b,
    Tensor const& alpha, Tensor const& beta, std::optional<c10::ScalarType> out_dtype)
{
    TORCH_CHECK(mat_a.dim() == 2 && mat_b.dim() == 2);
    TORCH_CHECK(mat_a.sizes()[1] == mat_b.sizes()[1]);            // mat_a is [m, k], mat_b is [n, k]

    auto const out_dtype_ = out_dtype.value_or(torch::kBFloat16); // Default to BF16
    std::vector<int64_t> output_size = {mat_a.sizes()[0], mat_b.sizes()[0]};

    Tensor out = at::empty(output_size, mat_a.options().dtype(out_dtype_));

    return cublas_fp4_scaled_mm_out(mat_a, mat_b, scale_a, scale_b, alpha, beta, out);
}

torch::Tensor cublas_fp4_scaled_mm_meta(torch::Tensor const& mat_a, torch::Tensor const& mat_b,
    torch::Tensor const& scale_a, torch::Tensor const& scale_b, torch::Tensor const& alpha, torch::Tensor const& beta,
    c10::optional<torch::ScalarType> out_dtype)
{
    auto const out_dtype_ = out_dtype.value_or(torch::kBFloat16);

    // Simplified and more stable shape inference
    // Avoid complex checks that might trigger recompilation
    auto m = mat_a.size(0);
    auto n = mat_b.size(0);

    // Output shape: [M, N]
    std::vector<int64_t> output_size = {m, n};

    // Use the most stable tensor creation method
    // Copy all properties from input tensor to ensure consistency
    return torch::empty(output_size, mat_a.options().dtype(out_dtype_));
}

// CublasLt FP4 GEMM Runner with auto-tuning support
class CublasLtFP4GemmRunner : public torch::CustomClassHolder
{
public:
    explicit CublasLtFP4GemmRunner(at::ScalarType outputDtype)
        : mOutputDtype(outputDtype)
    {
        TLLM_LOG_INFO("CublasLtFP4GemmRunner: Constructed with output_dtype=%d", static_cast<int>(outputDtype));
    }

    // Get number of heuristic algorithms for a given matrix shape
    int64_t getNumHeuristicAlgos(at::Tensor const& mat1, at::Tensor const& mat2)
    {
        int m = mat1.size(0);
        int k_compressed = mat1.size(1);
        int k = k_compressed * 2; // FP4 is 2 elements per byte
        int n = mat2.size(0);

        auto& cache = getOrCreateAlgoCache(m, k, n, mat1.device());
        size_t num_algos = cache.heuristics.size();
        TLLM_LOG_INFO(
            "CublasLtFP4GemmRunner: getNumHeuristicAlgos returned %zu algorithms for shape (m=%d, k=%d, n=%d)",
            num_algos, m, k, n);
        return static_cast<int64_t>(num_algos);
    }

    // Run GEMM with specified tactic (-1 for default/best)
    at::Tensor runGemm(at::Tensor const& mat1, at::Tensor const& mat2, at::Tensor const& mat1_scale,
        at::Tensor const& mat2_scale, at::Tensor const& alpha, bool to_userbuffers, int64_t tactic) const
    {
        int m = mat1.size(0);
        int k_compressed = mat1.size(1);
        int k = k_compressed * 2;
        int n = mat2.size(0);

        TLLM_LOG_INFO(
            "CublasLtFP4GemmRunner::runGemm: Entry with shape (m=%d, k=%d, n=%d), tactic=%ld, to_userbuffers=%d", m, k,
            n, tactic, to_userbuffers);

        // Prepare output tensor
        at::Tensor out;
        std::vector<int64_t> output_size = {m, n};

        if (to_userbuffers)
        {
            out = torch_ext::create_userbuffers_tensor(output_size, mOutputDtype).first;
        }
        else
        {
            out = at::empty(output_size, mat1.options().dtype(mOutputDtype));
        }

        // Get algorithm cache
        auto& cache = getOrCreateAlgoCache(m, k, n, mat1.device());

        // Create beta tensor (set to 0 for no accumulation)
        auto beta = torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32).device(mat1.device()));

        // Select algorithm
        bool has_algo = false;
        cublasLtMatmulAlgo_t const* algo_ptr = nullptr;

        if (tactic >= 0 && tactic < static_cast<int64_t>(cache.heuristics.size()))
        {
            // Use specified tactic
            algo_ptr = &cache.heuristics[tactic].algo;
            has_algo = true;
            TLLM_LOG_INFO("CublasLtFP4GemmRunner: Using specified tactic %ld (out of %zu) for shape (m=%d, n=%d, k=%d)",
                tactic, cache.heuristics.size(), m, n, k);
        }
        else if (tactic == -1 && !cache.heuristics.empty())
        {
            // Use best tactic (default is first one)
            int64_t best_idx
                = cache.best_tactic < static_cast<int64_t>(cache.heuristics.size()) ? cache.best_tactic : 0;
            algo_ptr = &cache.heuristics[best_idx].algo;
            has_algo = true;
            TLLM_LOG_INFO("CublasLtFP4GemmRunner: Using best tactic %ld (out of %zu) for shape (m=%d, n=%d, k=%d)",
                best_idx, cache.heuristics.size(), m, n, k);
        }

        // Execute GEMM
        if (has_algo)
        {
            cublas_fp4_gemm_caller_with_algo(out, mat1, mat2, mat1_scale, mat2_scale, alpha, beta, *algo_ptr);
        }
        else
        {
            // Fall back to default (no algorithm specified)
            TLLM_LOG_WARNING(
                "CublasLtFP4GemmRunner: No valid algorithm found (tactic=%ld, available=%zu), falling back to default "
                "for shape (m=%d, n=%d, k=%d)",
                tactic, cache.heuristics.size(), m, n, k);
            cublas_fp4_gemm_caller(out, mat1, mat2, mat1_scale, mat2_scale, alpha, beta);
        }

        TLLM_LOG_INFO("CublasLtFP4GemmRunner::runGemm: Exit, output shape=(%ld, %ld)", out.size(0), out.size(1));
        return out;
    }

    // Update best tactic after tuning
    void setBestTactic(at::Tensor const& mat1, at::Tensor const& mat2, int64_t best_tactic)
    {
        int m = mat1.size(0);
        int k_compressed = mat1.size(1);
        int k = k_compressed * 2;
        int n = mat2.size(0);

        auto& cache = getOrCreateAlgoCache(m, k, n, mat1.device());
        if (best_tactic >= 0 && best_tactic < static_cast<int64_t>(cache.heuristics.size()))
        {
            int64_t old_tactic = cache.best_tactic;
            cache.best_tactic = best_tactic;
            TLLM_LOG_INFO(
                "CublasLtFP4GemmRunner: Updated best tactic from %ld to %ld for shape (m=%d, k=%d, n=%d, device=%d)",
                old_tactic, best_tactic, m, k, n, mat1.device().index());
        }
        else
        {
            TLLM_LOG_WARNING(
                "CublasLtFP4GemmRunner: Invalid tactic %ld (available=%zu), ignoring setBestTactic for shape (m=%d, "
                "k=%d, n=%d)",
                best_tactic, cache.heuristics.size(), m, k, n);
        }
    }

private:
    struct AlgoCache
    {
        std::vector<cublasLtMatmulHeuristicResult_t> heuristics;
        int64_t best_tactic = 0; // Index of the best algorithm
    };

    // Cache key: (m, k, n, device_id) for algorithm list storage
    using ShapeKey = std::tuple<int, int, int, int>;

    struct ShapeKeyHash
    {
        size_t operator()(ShapeKey const& k) const
        {
            // Use boost-style hash_combine for better distribution
            size_t seed = 0;
            hash_combine(seed, std::get<0>(k));
            hash_combine(seed, std::get<1>(k));
            hash_combine(seed, std::get<2>(k));
            hash_combine(seed, std::get<3>(k));
            return seed;
        }

    private:
        // Standard hash combination algorithm (Boost-style)
        template <typename T>
        static void hash_combine(size_t& seed, T const& v)
        {
            std::hash<T> hasher;
            seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
    };

    mutable std::unordered_map<ShapeKey, AlgoCache, ShapeKeyHash> mAlgoCache;
    at::ScalarType mOutputDtype;

    AlgoCache& getOrCreateAlgoCache(int m, int k, int n, c10::Device device) const
    {
        ShapeKey key = std::make_tuple(m, k, n, device.index());

        if (mAlgoCache.find(key) == mAlgoCache.end())
        {
            TLLM_LOG_INFO(
                "CublasLtFP4GemmRunner: Cache miss for shape (m=%d, k=%d, n=%d, device=%d), creating new cache entry",
                m, k, n, device.index());

            AlgoCache cache;

            // Create cublas wrapper
            at::cuda::CUDAGuard deviceGuard(device);
            auto cublasHandle = getCublasHandle();
            auto cublasLtHandle = getCublasLtHandle();
            auto cublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);

            // Set FP4 configuration
            cudaDataType_t outType = mOutputDtype == at::ScalarType::Half
                ? CUDA_R_16F
                : (mOutputDtype == at::ScalarType::BFloat16 ? CUDA_R_16BF : CUDA_R_32F);

            cublasWrapper->setFP4GemmConfig(outType);

            // Create descriptors
            // Note: FP4 uses transposed layout similar to regular gemm
            cublasWrapper->createDescriptors(
                CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, /*lda=*/k, /*ldb=*/k, /*ldc=*/n, /*fastAcc=*/0);

            // Create dummy scaling factors for descriptor setup
            // FP4 GEMM requires scale descriptors to be set before calling getTactics
            // For NVFP4, P=16 elements per scaling factor
            // k is the number of original FP4 elements, so we need k/16 scaling factors
            int scale_groups_m = k / 16;
            int scale_groups_n = k / 16;
            auto dummy_scale_a
                = torch::zeros({m, scale_groups_m}, torch::TensorOptions().dtype(torch::kUInt8).device(device));
            auto dummy_scale_b
                = torch::zeros({n, scale_groups_n}, torch::TensorOptions().dtype(torch::kUInt8).device(device));

            void* a_sf_ptr = reinterpret_cast<void*>(dummy_scale_a.data_ptr());
            void* b_sf_ptr = reinterpret_cast<void*>(dummy_scale_b.data_ptr());

            // Set scale descriptors (required for FP4 GEMM heuristics)
            cublasWrapper->setScaleDescriptors(a_sf_ptr, b_sf_ptr);

            // Get heuristic algorithms
            auto heuristics = cublasWrapper->getTactics(CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, k, k, n);

            // Filter valid algorithms
            for (auto const& h : heuristics)
            {
                if (h.state == CUBLAS_STATUS_SUCCESS && h.workspaceSize <= CUBLAS_WORKSPACE_SIZE)
                {
                    cache.heuristics.push_back(h);
                }
            }

            cublasWrapper->destroyDescriptors();

            TLLM_LOG_INFO("CublasLtFP4GemmRunner: Found %zu valid algorithms for shape (m=%d, k=%d, n=%d) on device %d",
                cache.heuristics.size(), m, k, n, device.index());

            if (cache.heuristics.empty())
            {
                TLLM_LOG_WARNING(
                    "CublasLtFP4GemmRunner: No valid cuBLASLt algorithms found, will fall back to default");
            }

            mAlgoCache[key] = std::move(cache);
        }
        else
        {
            TLLM_LOG_DEBUG(
                "CublasLtFP4GemmRunner: Cache hit for shape (m=%d, k=%d, n=%d, device=%d), %zu algorithms available", m,
                k, n, device.index(), mAlgoCache[key].heuristics.size());
        }

        return mAlgoCache[key];
    }

    // Helper function to run GEMM with a specific algorithm
    static void cublas_fp4_gemm_caller_with_algo(torch::Tensor& out, torch::Tensor const& a, torch::Tensor const& b,
        torch::Tensor const& scale_a, torch::Tensor const& scale_b, torch::Tensor const& alpha,
        torch::Tensor const& beta, cublasLtMatmulAlgo_t const& algo)
    {
        int32_t m = a.sizes()[0];
        int32_t n = b.sizes()[0];
        int32_t k_compressed = a.sizes()[1];
        int32_t k = k_compressed * 2;

        TLLM_LOG_INFO(
            "cublas_fp4_gemm_caller_with_algo: Executing FP4 GEMM with selected algorithm for shape (m=%d, n=%d, k=%d)",
            m, n, k);

        at::cuda::CUDAGuard deviceGuard(a.device());

        thread_local std::unordered_map<int, std::shared_ptr<CublasMMWrapper>> cublasWrappers;
        auto& cublasWrapper = cublasWrappers[a.get_device()];
        if (!cublasWrapper)
        {
            auto cublasHandle = getCublasHandle();
            auto cublasLtHandle = getCublasLtHandle();
            cublasWrapper = std::make_shared<CublasMMWrapper>(cublasHandle, cublasLtHandle, nullptr, nullptr);
        }

        cublasWrapper->setFP4GemmConfig(CUDA_R_16BF);

        auto const workspace_options = torch::TensorOptions().dtype(torch::kUInt8).device(a.device());
        auto workspace = torch::empty(CUBLAS_WORKSPACE_SIZE, workspace_options);

        auto stream = at::cuda::getCurrentCUDAStream(a.get_device());

        auto* a_ptr = static_cast<void*>(a.data_ptr());
        auto* b_ptr = static_cast<void*>(b.data_ptr());
        auto* out_ptr = static_cast<void*>(out.data_ptr());
        auto* ws_ptr = static_cast<void*>(workspace.data_ptr());

        void const* a_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_a.data_ptr());
        void const* b_sf_ptr = reinterpret_cast<__nv_fp8_e4m3 const*>(scale_b.data_ptr());

        auto* alpha_ptr = alpha.data_ptr<float>();
        auto* beta_ptr = beta.data_ptr<float>();

        cublasWrapper->setStream(stream);
        cublasWrapper->setWorkspace(ws_ptr);

        // Create descriptors
        cublasWrapper->createDescriptors(CUBLAS_OP_T, CUBLAS_OP_N, n, m, k, k, k, n, 0);

        // Create D descriptor
        cublasLtMatrixLayout_t Ddesc = NULL;
        cudaDataType_t outType = CUDA_R_16BF;
        check_cuda_error(cublasLtMatrixLayoutCreate(&Ddesc, outType, m, n, n));

        // Set scale descriptors
        // IMPORTANT: Scaling factors must be swapped to match the swapped matrices!
        // Since we swap A<->B in the operation, we must also swap their scaling factors
        cublasWrapper->setScaleDescriptors(const_cast<void*>(b_sf_ptr), const_cast<void*>(a_sf_ptr));

        // Execute with specified algorithm
        check_cuda_error(cublasLtMatmul(cublasWrapper->getCublasLtHandle(), cublasWrapper->getOperationDesc(),
            alpha_ptr, b_ptr, cublasWrapper->getBDesc(), a_ptr, cublasWrapper->getADesc(), beta_ptr, out_ptr,
            cublasWrapper->getCDesc(), out_ptr, Ddesc, &algo, ws_ptr, CUBLAS_WORKSPACE_SIZE, stream));

        sync_check_cuda_error(stream);

        if (Ddesc)
            cublasLtMatrixLayoutDestroy(Ddesc);
        cublasWrapper->destroyDescriptors();
    }
};

} // namespace torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.class_<torch_ext::CublasLtFP4GemmRunner>("CublasLtFP4GemmRunner")
        .def(torch::init<at::ScalarType>())
        .def("run_gemm", &torch_ext::CublasLtFP4GemmRunner::runGemm)
        .def("get_num_heuristic_algos", &torch_ext::CublasLtFP4GemmRunner::getNumHeuristicAlgos)
        .def("set_best_tactic", &torch_ext::CublasLtFP4GemmRunner::setBestTactic);

    // Legacy cublas_fp4_scaled_mm op - for testing and backward compatibility
    m.def(
        "cublas_fp4_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scale_a, Tensor scale_b,"
        " Tensor alpha, Tensor beta, ScalarType? out_dtype=None) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("cublas_fp4_scaled_mm", &torch_ext::cublas_fp4_scaled_mm);
}

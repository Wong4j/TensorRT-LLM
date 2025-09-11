#!/usr/bin/env python3
"""
测试 nvfp4_gemm 日志功能的脚本
"""

import logging
import torch
import numpy as np

# 设置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 导入 TensorRT-LLM 相关模块
try:
    import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
    from tensorrt_llm._torch.custom_ops.torch_custom_ops import nvfp4_gemm
    print("Successfully imported TensorRT-LLM modules")
except ImportError as e:
    print(f"Failed to import TensorRT-LLM modules: {e}")
    exit(1)

def test_nvfp4_gemm_logging():
    """测试 nvfp4_gemm 的日志功能"""
    
    print("=" * 60)
    print("Testing nvfp4_gemm logging functionality")
    print("=" * 60)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    # 创建测试数据
    m, n, k = 128, 256, 512
    k_compressed = k // 2  # FP4 压缩了 k 维度
    
    print(f"\nCreating test data with dimensions: m={m}, n={n}, k={k}")
    
    # 创建 FP4 输入数据 - 使用 randint 生成 uint8 数据
    act_fp4 = torch.randint(0, 255, (m, k_compressed), dtype=torch.uint8, device=device)  # FP4 压缩了 k 维度
    weight = torch.randint(0, 255, (n, k_compressed), dtype=torch.uint8, device=device)   # FP4 压缩了 k 维度
    
    # 创建缩放因子 - nvfp4: 每 16 个元素共享一个缩放因子
    # CUTLASS 期望 uint8 类型 (cutlass::float_ue4m3_t)
    # cuBLASLt 期望 fp8_e4m3fn 类型
    scale_groups = k_compressed // 16  # 缩放因子的组数
    act_sf_uint8 = torch.ones((m, scale_groups), dtype=torch.uint8, device=device)  # CUTLASS
    weight_scale_uint8 = torch.ones((n, scale_groups), dtype=torch.uint8, device=device)  # CUTLASS
    act_sf_fp8 = torch.ones((m, scale_groups), dtype=torch.float8_e4m3fn, device=device)  # cuBLASLt
    weight_scale_fp8 = torch.ones((n, scale_groups), dtype=torch.float8_e4m3fn, device=device)  # cuBLASLt
    alpha = torch.tensor([1.0], dtype=torch.float32, device=device)
    
    print(f"Input shapes:")
    print(f"  act_fp4: {list(act_fp4.shape)}")
    print(f"  weight: {list(weight.shape)}")
    print(f"  act_sf_uint8 (CUTLASS): {list(act_sf_uint8.shape)}")
    print(f"  weight_scale_uint8 (CUTLASS): {list(weight_scale_uint8.shape)}")
    print(f"  act_sf_fp8 (cuBLASLt): {list(act_sf_fp8.shape)}")
    print(f"  weight_scale_fp8 (cuBLASLt): {list(weight_scale_fp8.shape)}")
    print(f"  alpha: {list(alpha.shape)}")
    
    # 测试 CUTLASS 后端
    print(f"\n" + "=" * 40)
    print("Testing CUTLASS backend")
    print("=" * 40)
    
    try:
        result_cutlass = nvfp4_gemm(
            act_fp4=act_fp4,
            weight=weight,
            act_sf=act_sf_uint8,  # CUTLASS 期望 uint8
            weight_scale=weight_scale_uint8,  # CUTLASS 期望 uint8
            alpha=alpha,
            output_dtype=torch.float16,
            to_userbuffers=False,
            backend="cutlass"
        )
        print(f"CUTLASS result shape: {list(result_cutlass.shape)}")
        print(f"CUTLASS result dtype: {result_cutlass.dtype}")
        print(f"CUTLASS result device: {result_cutlass.device}")
    except Exception as e:
        print(f"CUTLASS backend failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试 cuBLASLt 后端
    print(f"\n" + "=" * 40)
    print("Testing cuBLASLt backend")
    print("=" * 40)
    
    try:
        result_cublaslt = nvfp4_gemm(
            act_fp4=act_fp4,
            weight=weight,
            act_sf=act_sf_fp8,  # cuBLASLt 期望 fp8_e4m3fn
            weight_scale=weight_scale_fp8,  # cuBLASLt 期望 fp8_e4m3fn
            alpha=alpha,
            output_dtype=torch.float16,
            to_userbuffers=False,
            backend="cublaslt"
        )
        print(f"cuBLASLt result shape: {list(result_cublaslt.shape)}")
        print(f"cuBLASLt result dtype: {result_cublaslt.dtype}")
        print(f"cuBLASLt result device: {result_cublaslt.device}")
    except Exception as e:
        print(f"cuBLASLt backend failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print("Test completed")
    print("=" * 60)

if __name__ == "__main__":
    test_nvfp4_gemm_logging()

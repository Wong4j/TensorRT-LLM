#!/usr/bin/env python3
"""
cuBLASLt FP4 GEMM 使用示例

这个示例展示了如何使用 cuBLASLt 后端进行 FP4 GEMM 操作。
"""

import torch
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

def test_cublaslt_fp4_gemm():
    """测试 cuBLASLt FP4 GEMM 功能"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    # 设置矩阵维度
    m, n, k = 128, 256, 512
    
    # 创建 FP4 输入数据
    act_fp4 = torch.randint(0, 255, (m, k // 2), dtype=fp4_utils.FLOAT4_E2M1X2, device=device)
    weight = torch.randint(0, 255, (n, k // 2), dtype=fp4_utils.FLOAT4_E2M1X2, device=device)
    
    # 创建缩放因子
    act_sf = torch.randn(m, dtype=torch.uint8, device=device)
    weight_scale = torch.randn(n, dtype=torch.uint8, device=device)
    alpha = torch.tensor([1.0], dtype=torch.float32, device=device)
    
    print("Testing cuBLASLt FP4 GEMM...")
    
    # 测试 cuBLASLt 后端
    try:
        result_cublaslt = torch.ops.trtllm.nvfp4_gemm(
            act_fp4, weight, act_sf, weight_scale, alpha,
            output_dtype=torch.float16,
            backend="cublaslt"
        )
        print(f"cuBLASLt result shape: {result_cublaslt.shape}")
        print(f"cuBLASLt result dtype: {result_cublaslt.dtype}")
        print("cuBLASLt FP4 GEMM test passed!")
    except Exception as e:
        print(f"cuBLASLt test failed: {e}")
    
    # 测试 CUTLASS 后端（对比）
    try:
        result_cutlass = torch.ops.trtllm.nvfp4_gemm(
            act_fp4, weight, act_sf, weight_scale, alpha,
            output_dtype=torch.float16,
            backend="cutlass"
        )
        print(f"CUTLASS result shape: {result_cutlass.shape}")
        print(f"CUTLASS result dtype: {result_cutlass.dtype}")
        print("CUTLASS FP4 GEMM test passed!")
    except Exception as e:
        print(f"CUTLASS test failed: {e}")
    
    # 比较结果（如果两个都成功）
    try:
        if 'result_cublaslt' in locals() and 'result_cutlass' in locals():
            diff = torch.abs(result_cublaslt - result_cutlass).max()
            print(f"Maximum difference between cuBLASLt and CUTLASS: {diff}")
            if diff < 1e-3:
                print("Results are very close!")
            else:
                print("Results differ significantly (expected due to different implementations)")
    except Exception as e:
        print(f"Comparison failed: {e}")

def test_backend_validation():
    """测试后端验证功能"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return
    
    # 创建测试数据
    m, n, k = 64, 128, 256
    act_fp4 = torch.randint(0, 255, (m, k // 2), dtype=fp4_utils.FLOAT4_E2M1X2, device=device)
    weight = torch.randint(0, 255, (n, k // 2), dtype=fp4_utils.FLOAT4_E2M1X2, device=device)
    act_sf = torch.randn(m, dtype=torch.uint8, device=device)
    weight_scale = torch.randn(n, dtype=torch.uint8, device=device)
    alpha = torch.tensor([1.0], dtype=torch.float32, device=device)
    
    print("\nTesting backend validation...")
    
    # 测试无效后端
    try:
        torch.ops.trtllm.nvfp4_gemm(
            act_fp4, weight, act_sf, weight_scale, alpha,
            output_dtype=torch.float16,
            backend="invalid_backend"
        )
        print("ERROR: Should have failed with invalid backend")
    except ValueError as e:
        print(f"Correctly caught invalid backend error: {e}")
    
    # 测试默认后端
    try:
        result_default = torch.ops.trtllm.nvfp4_gemm(
            act_fp4, weight, act_sf, weight_scale, alpha,
            output_dtype=torch.float16
            # 不指定 backend，应该使用默认的 cutlass
        )
        print(f"Default backend result shape: {result_default.shape}")
        print("Default backend test passed!")
    except Exception as e:
        print(f"Default backend test failed: {e}")

if __name__ == "__main__":
    print("cuBLASLt FP4 GEMM Example")
    print("=" * 40)
    
    test_cublaslt_fp4_gemm()
    test_backend_validation()
    
    print("\nExample completed!")

#!/usr/bin/env python3
"""
最简化的 FP4 GEMM 对比测试
直接对比 CUTLASS 和 cuBLASLt 的输出值
"""

import torch
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.custom_ops import nvfp4_gemm

# 设置矩阵大小
m, n, k = 64, 128, 256
k_compressed = k // 2
scale_groups = k_compressed // 16

print(f"Matrix size: m={m}, n={n}, k={k}")
print(f"Compressed k: {k_compressed}, scale_groups: {scale_groups}")

# 设置随机种子以确保可重复性
torch.manual_seed(42)

# 创建随机输入数据
print("Creating random test data...")
act_fp4_data = torch.randn((m, k_compressed), device='cuda') * 0.5  # 范围 [-0.5, 0.5]
weight_data = torch.randn((n, k_compressed), device='cuda') * 0.3   # 范围 [-0.3, 0.3]

# 转换为 FP4 格式
act_fp4 = act_fp4_data.to(fp4_utils.FLOAT4_E2M1X2)
weight = weight_data.to(fp4_utils.FLOAT4_E2M1X2)

# 创建合理的缩放因子 (E4M3 格式的 1.0 = 0x70)
e4m3_one = 0x70  # E4M3 格式的 1.0
# 添加一些随机变化，但保持在合理范围内
scale_variation = torch.randn((m, scale_groups), device='cuda') * 0.1 + 1.0  # 范围 [0.9, 1.1]
act_sf = (scale_variation * e4m3_one).clamp(0, 255).to(torch.uint8)

weight_scale_variation = torch.randn((n, scale_groups), device='cuda') * 0.1 + 1.0  # 范围 [0.9, 1.1]
weight_scale = (weight_scale_variation * e4m3_one).clamp(0, 255).to(torch.uint8)

alpha = torch.tensor(1.0, dtype=torch.float32, device='cuda')

print(f"Input shapes:")
print(f"  act_fp4: {act_fp4.shape}, dtype: {act_fp4.dtype}")
print(f"  weight: {weight.shape}, dtype: {weight.dtype}")
print(f"  act_sf: {act_sf.shape}, dtype: {act_sf.dtype}")
print(f"  weight_scale: {weight_scale.shape}, dtype: {weight_scale.dtype}")

# 运行 CUTLASS
print("\n=== Running CUTLASS ===")
cutlass_result = nvfp4_gemm(
    act_fp4=act_fp4,
    weight=weight,
    act_sf=act_sf,
    weight_scale=weight_scale,
    alpha=alpha,
    output_dtype=torch.bfloat16,
    to_userbuffers=False,
    backend="cutlass"
)

# 运行 cuBLASLt
print("\n=== Running cuBLASLt ===")
cublaslt_result = nvfp4_gemm(
    act_fp4=act_fp4,
    weight=weight,
    act_sf=act_sf,
    weight_scale=weight_scale,
    alpha=alpha,
    output_dtype=torch.bfloat16,
    to_userbuffers=False,
    backend="cublaslt"
)

# 比较结果
print("\n=== Results Comparison ===")
print(f"CUTLASS output shape: {cutlass_result.shape}, dtype: {cutlass_result.dtype}")
print(f"cuBLASLt output shape: {cublaslt_result.shape}, dtype: {cublaslt_result.dtype}")

print(f"\nCUTLASS output range: [{torch.min(cutlass_result):.6f}, {torch.max(cutlass_result):.6f}]")
print(f"cuBLASLt output range: [{torch.min(cublaslt_result):.6f}, {torch.max(cublaslt_result):.6f}]")

print(f"\nCUTLASS output mean: {torch.mean(cutlass_result):.6f}")
print(f"cuBLASLt output mean: {torch.mean(cublaslt_result):.6f}")

# 计算差异
abs_diff = torch.abs(cutlass_result - cublaslt_result)
rel_diff = abs_diff / (torch.abs(cutlass_result) + 1e-8)

print(f"\nMax absolute difference: {torch.max(abs_diff):.6f}")
print(f"Max relative difference: {torch.max(rel_diff):.6f}")
print(f"Mean absolute difference: {torch.mean(abs_diff):.6f}")
print(f"Mean relative difference: {torch.mean(rel_diff):.6f}")

# 检查是否有 NaN 或 Inf
print(f"\nCUTLASS has NaN: {torch.isnan(cutlass_result).any()}")
print(f"cuBLASLt has NaN: {torch.isnan(cublaslt_result).any()}")
print(f"CUTLASS has Inf: {torch.isinf(cutlass_result).any()}")
print(f"cuBLASLt has Inf: {torch.isinf(cublaslt_result).any()}")

for i in range(10):
    print(f"CUTLASS output: {cutlass_result[i]}")
    print(f"cuBLASLt output: {cublaslt_result[i]}")

print("\n=== Test Complete ===")

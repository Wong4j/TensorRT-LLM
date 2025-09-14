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

# 创建随机输入数据 - 使用适中的范围确保 FP4 格式有效且有非零输出
print("Creating random test data...")
# 使用离散的整数值，避免浮点精度问题
act_fp4_data = torch.randint(-10, 11, (m, k_compressed), device='cuda').float() * 0.1  # 范围 [-1.0, 1.0]
weight_data = torch.randint(-10, 11, (n, k_compressed), device='cuda').float() * 0.1   # 范围 [-1.0, 1.0]

# 转换为 FP4 格式
act_fp4 = act_fp4_data.to(fp4_utils.FLOAT4_E2M1X2)
weight = weight_data.to(fp4_utils.FLOAT4_E2M1X2)

# 验证 FP4 数据是否有效
if torch.isnan(act_fp4.float()).any() or torch.isnan(weight.float()).any():
    print("Warning: Detected NaN in FP4 data, using zeros instead")
    act_fp4 = torch.zeros((m, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device='cuda')
    weight = torch.zeros((n, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device='cuda')

# 创建缩放因子 - 使用固定的 E4M3 格式值，确保数值稳定性
e4m3_one = 0x70  # E4M3 格式的 1.0
# 使用固定的缩放因子，避免随机变化导致无效值
act_sf = torch.full((m, scale_groups), e4m3_one, dtype=torch.uint8, device='cuda')
weight_scale = torch.full((n, scale_groups), e4m3_one, dtype=torch.uint8, device='cuda')

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

# 添加 allclose 对比
allclose_result = torch.allclose(cutlass_result, cublaslt_result, rtol=1e-5, atol=1e-8)
strict_allclose = torch.allclose(cutlass_result, cublaslt_result, rtol=1e-6, atol=1e-9)

element_wise_equal = torch.isclose(cutlass_result, cublaslt_result, rtol=1e-5, atol=1e-8)
equal_count = torch.sum(element_wise_equal).item()
total_count = element_wise_equal.numel()
equal_percentage = (equal_count / total_count) * 100

print(f"\nAllclose (rtol=1e-5, atol=1e-8): {allclose_result}")
print(f"Strict allclose (rtol=1e-6, atol=1e-9): {strict_allclose}")
print(f"Equal elements: {equal_count}/{total_count} ({equal_percentage:.2f}%)")

# 检查是否有 NaN 或 Inf
print(f"\nCUTLASS has NaN: {torch.isnan(cutlass_result).any()}")
print(f"cuBLASLt has NaN: {torch.isnan(cublaslt_result).any()}")
print(f"CUTLASS has Inf: {torch.isinf(cutlass_result).any()}")
print(f"cuBLASLt has Inf: {torch.isinf(cublaslt_result).any()}")

for i in range(10):
    print(f"CUTLASS output: {cutlass_result[i]}")
    print(f"cuBLASLt output: {cublaslt_result[i]}")

print("\n=== Test Complete ===")

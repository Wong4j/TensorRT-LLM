#!/usr/bin/env python3
"""
调试 262144 这个数值的来源
"""

import torch
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

# 设置矩阵大小
m, n, k = 64, 128, 256
k_compressed = k // 2
scale_groups = k_compressed // 16

print(f"Matrix dimensions: m={m}, n={n}, k={k}")
print(f"Compressed k: {k_compressed}, scale_groups: {scale_groups}")

# 理论计算
print(f"\n=== Theoretical Calculation ===")
print(f"Matrix A shape: [{m}, {k_compressed}]")
print(f"Matrix B shape: [{n}, {k_compressed}]")
print(f"Each output element = sum of {k_compressed} products")
print(f"With all-ones matrices: {k_compressed} × 1 × 1 = {k_compressed}")

# 检查 FP4 的隐式缩放
print(f"\n=== FP4 Implicit Scaling ===")
print(f"262144 = 2^18 = {2**18}")
print(f"262144 / {k_compressed} = {262144 / k_compressed}")
print(f"262144 / {k} = {262144 / k}")

# 检查是否是 2 的幂次
print(f"\n=== Power of 2 Analysis ===")
exp = 262144.bit_length() - 1
print(f"262144 = 2^{exp}")
k_exp = k.bit_length() - 1
print(f"k = {k} = 2^{k_exp}")
k_comp_exp = k_compressed.bit_length() - 1
print(f"k_compressed = {k_compressed} = 2^{k_comp_exp}")

# 检查可能的缩放因子
print(f"\n=== Possible Scaling Factors ===")
possible_factors = [2**i for i in range(10, 20)]
for factor in possible_factors:
    if 262144 % factor == 0:
        quotient = 262144 // factor
        factor_exp = factor.bit_length() - 1
        print(f"262144 = {factor} × {quotient} = 2^{factor_exp} × {quotient}")

# 检查与矩阵维度的关系
print(f"\n=== Matrix Dimension Relationships ===")
print(f"262144 / m = {262144 / m}")
print(f"262144 / n = {262144 / n}")
print(f"262144 / k = {262144 / k}")
print(f"262144 / k_compressed = {262144 / k_compressed}")

# 检查是否是某个常数的倍数
print(f"\n=== Constant Multipliers ===")
constants = [128, 256, 512, 1024, 2048, 4096]
for const in constants:
    if 262144 % const == 0:
        print(f"262144 = {const} × {262144 // const}")

print(f"\n=== Conclusion ===")
print(f"262144 = 2^18 = 256 × 1024")
print(f"This suggests the calculation involves:")
print(f"  - k dimension (256) as a scaling factor")
print(f"  - Some internal scaling factor (1024 = 2^10)")
print(f"  - Or possibly: 256 × 1024 = k × 2^10")

#!/usr/bin/env python3
"""
测试scaling factor类型转换问题
"""

import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_scaling_factor_conversion():
    """测试scaling factor的类型转换"""
    logger.info("=" * 60)
    logger.info("Testing scaling factor type conversion")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available, skipping test")
        return False
    
    # 创建测试数据
    device = "cuda"
    m, n, k = 64, 128, 256  # 使用与之前测试相同的尺寸
    k_compressed = k // 2
    scale_groups = k_compressed // 16
    
    logger.info(f"Matrix dimensions: m={m}, n={n}, k={k}")
    logger.info(f"Compressed k: {k_compressed}, scale_groups: {scale_groups}")
    
    # 创建全为1的矩阵
    act_fp4 = torch.ones((m, k_compressed), dtype=torch.uint8, device=device)
    weight = torch.ones((n, k_compressed), dtype=torch.uint8, device=device)
    
    # 测试不同的scaling factor值
    # 使用float8_e4m3fn格式的bit位表示
    # 1.0 = 0x70 (0111 0000)
    # 2.0 = 0x78 (0111 1000) 
    # 0.5 = 0x68 (0110 1000)
    # 0.25 = 0x60 (0110 0000)
    # 0.125 = 0x58 (0101 1000)
    test_values = [0x70, 0x78, 0x68, 0x60, 0x58, 0x50, 0x48, 0x40, 0x38]
    
    # 定义float8_e4m3fn bit位到浮点数值的映射
    fp8_values = {
        0x70: 1.0,    # 0111 0000
        0x78: 2.0,    # 0111 1000
        0x68: 0.5,    # 0110 1000
        0x60: 0.25,   # 0110 0000
        0x58: 0.125,  # 0101 1000
        0x50: 0.0625, # 0101 0000
        0x48: 0.03125,# 0100 1000
        0x40: 0.015625,# 0100 0000
        0x38: 0.0078125 # 0011 1000
    }
    
    for val in test_values:
        fp8_val = fp8_values.get(val, "unknown")
        logger.info(f"\n--- Testing scaling factor value: 0x{val:02x} ({fp8_val}) ---")
        
        # 创建scaling factor
        act_sf = torch.full((m, scale_groups), val, dtype=torch.uint8, device=device)
        weight_scale = torch.full((n, scale_groups), val, dtype=torch.uint8, device=device)
        alpha = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        logger.info(f"act_sf shape: {list(act_sf.shape)}, dtype: {act_sf.dtype}")
        logger.info(f"act_sf values (hex): {[hex(x) for x in act_sf[0, :].cpu().numpy()]}")
        
        # 测试CUTLASS
        try:
            from tensorrt_llm._torch.custom_ops import nvfp4_gemm
            
            result_cutlass = nvfp4_gemm(
                act_fp4=act_fp4,
                weight=weight,
                act_sf=act_sf,
                weight_scale=weight_scale,
                alpha=alpha,
                output_dtype=torch.bfloat16,
                to_userbuffers=False,
                backend="cutlass"
            )
            
            logger.info(f"CUTLASS result range: [{torch.min(result_cutlass):.6f}, {torch.max(result_cutlass):.6f}]")
            logger.info(f"CUTLASS result mean: {torch.mean(result_cutlass):.6f}")
            
        except Exception as e:
            logger.error(f"CUTLASS failed: {e}")
            continue
        
        # 测试cuBLASLt
        try:
            result_cublaslt = nvfp4_gemm(
                act_fp4=act_fp4,
                weight=weight,
                act_sf=act_sf,
                weight_scale=weight_scale,
                alpha=alpha,
                output_dtype=torch.bfloat16,
                to_userbuffers=False,
                backend="cublaslt"
            )
            
            logger.info(f"cuBLASLt result range: [{torch.min(result_cublaslt):.6f}, {torch.max(result_cublaslt):.6f}]")
            logger.info(f"cuBLASLt result mean: {torch.mean(result_cublaslt):.6f}")
            
            # 比较结果
            abs_diff = torch.abs(result_cutlass - result_cublaslt)
            max_diff = torch.max(abs_diff).item()
            mean_diff = torch.mean(abs_diff).item()
            
            logger.info(f"Max difference: {max_diff:.6f}")
            logger.info(f"Mean difference: {mean_diff:.6f}")
            
            # 计算比例
            if torch.mean(result_cutlass) > 0:
                ratio = torch.mean(result_cublaslt) / torch.mean(result_cutlass)
                logger.info(f"cuBLASLt/CUTLASS ratio: {ratio:.6f}")
            
        except Exception as e:
            logger.error(f"cuBLASLt failed: {e}")
            continue

if __name__ == "__main__":
    test_scaling_factor_conversion()

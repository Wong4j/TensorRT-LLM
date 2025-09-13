#!/usr/bin/env python3
"""
调试scaling factor的数值关系
"""

import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_scaling_factor():
    """调试scaling factor的数值关系"""
    logger.info("=" * 60)
    logger.info("Debugging scaling factor numerical relationship")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available, skipping test")
        return False
    
    # 测试参数
    device = "cuda"
    m, n, k = 64, 128, 256
    k_compressed = k // 2
    scale_groups = k_compressed // 16
    
    logger.info(f"Matrix dimensions: m={m}, n={n}, k={k}")
    logger.info(f"Compressed k: {k_compressed}, scale_groups: {scale_groups}")
    
    # 创建全为1的矩阵
    act_fp4 = torch.ones((m, k_compressed), dtype=torch.uint8, device=device)
    weight = torch.ones((n, k_compressed), dtype=torch.uint8, device=device)
    
    # 测试不同的scaling factor值
    test_scales = [
        (0x70, 1.0),    # float8_e4m3fn格式的1.0
        (0x78, 2.0),    # float8_e4m3fn格式的2.0
        (0x68, 0.5),    # float8_e4m3fn格式的0.5
        (0x60, 0.25),   # float8_e4m3fn格式的0.25
    ]
    
    for scale_val, scale_float in test_scales:
        logger.info(f"\n--- Testing scaling factor: 0x{scale_val:02x} ({scale_float}) ---")
        
        # 创建scaling factor
        act_sf = torch.full((m, scale_groups), scale_val, dtype=torch.uint8, device=device)
        weight_scale = torch.full((n, scale_groups), scale_val, dtype=torch.uint8, device=device)
        alpha = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        # 测试CUTLASS
        try:
            from tensorrt_llm._torch.custom_ops import nvfp4_gemm
            
            result = nvfp4_gemm(
                act_fp4=act_fp4,
                weight=weight,
                act_sf=act_sf,
                weight_scale=weight_scale,
                alpha=alpha,
                output_dtype=torch.bfloat16,
                to_userbuffers=False,
                backend="cutlass"
            )
            
            result_val = result[0, 0].item()
            logger.info(f"CUTLASS result: {result_val}")
            
            # 分析数值关系
            if scale_float == 1.0:
                base_result = result_val
                logger.info(f"Base result (scale=1.0): {base_result}")
            else:
                ratio = result_val / base_result if 'base_result' in locals() else 1.0
                expected_ratio = scale_float * scale_float  # 两个scaling factor相乘
                logger.info(f"Result ratio: {ratio:.6f}")
                logger.info(f"Expected ratio: {expected_ratio:.6f}")
                logger.info(f"Ratio match: {abs(ratio - expected_ratio) < 0.01}")
                
        except Exception as e:
            logger.error(f"CUTLASS failed: {e}")
            continue

if __name__ == "__main__":
    debug_scaling_factor()

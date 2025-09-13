#!/usr/bin/env python3
"""
验证FP4 GEMM的理论结果
"""

import torch
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_theoretical_result():
    """验证FP4 GEMM的理论结果"""
    logger.info("=" * 60)
    logger.info("Testing theoretical FP4 GEMM result")
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
    
    # 创建scaling factor = 1 (float8_e4m3fn格式: 0x70)
    scale_val = 0x70  # float8_e4m3fn格式的1.0
    act_sf = torch.full((m, scale_groups), scale_val, dtype=torch.uint8, device=device)
    weight_scale = torch.full((n, scale_groups), scale_val, dtype=torch.uint8, device=device)
    alpha = torch.tensor(1.0, dtype=torch.float32, device=device)
    
    logger.info(f"act_fp4 shape: {list(act_fp4.shape)}, dtype: {act_fp4.dtype}")
    logger.info(f"weight shape: {list(weight.shape)}, dtype: {weight.dtype}")
    logger.info(f"act_sf shape: {list(act_sf.shape)}, dtype: {act_sf.dtype}")
    logger.info(f"weight_scale shape: {list(weight_scale.shape)}, dtype: {weight_scale.dtype}")
    logger.info(f"alpha: {alpha.item()}")
    
    # 理论结果计算
    # 对于全1矩阵，每个输出元素应该是 k * scale_A * scale_B
    # 由于FP4压缩，实际k是k_compressed
    theoretical_result = k_compressed * 1.0 * 1.0  # scale都是1
    logger.info(f"Theoretical result (each element): {theoretical_result}")
    logger.info(f"Theoretical result (k_compressed): {k_compressed}")
    
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
        
        logger.info(f"CUTLASS result shape: {list(result_cutlass.shape)}")
        logger.info(f"CUTLASS result range: [{torch.min(result_cutlass):.6f}, {torch.max(result_cutlass):.6f}]")
        logger.info(f"CUTLASS result mean: {torch.mean(result_cutlass):.6f}")
        logger.info(f"CUTLASS result std: {torch.std(result_cutlass):.6f}")
        
        # 检查是否接近理论值
        diff = torch.abs(result_cutlass - theoretical_result)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        logger.info(f"CUTLASS vs theoretical - Max diff: {max_diff:.6f}")
        logger.info(f"CUTLASS vs theoretical - Mean diff: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            logger.info("✅ CUTLASS result matches theoretical expectation")
        else:
            logger.warning("⚠️ CUTLASS result differs from theoretical expectation")
            
    except Exception as e:
        logger.error(f"CUTLASS failed: {e}")
    
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
        
        logger.info(f"cuBLASLt result shape: {list(result_cublaslt.shape)}")
        logger.info(f"cuBLASLt result range: [{torch.min(result_cublaslt):.6f}, {torch.max(result_cublaslt):.6f}]")
        logger.info(f"cuBLASLt result mean: {torch.mean(result_cublaslt):.6f}")
        logger.info(f"cuBLASLt result std: {torch.std(result_cublaslt):.6f}")
        
        # 检查是否接近理论值
        diff = torch.abs(result_cublaslt - theoretical_result)
        max_diff = torch.max(diff).item()
        mean_diff = torch.mean(diff).item()
        
        logger.info(f"cuBLASLt vs theoretical - Max diff: {max_diff:.6f}")
        logger.info(f"cuBLASLt vs theoretical - Mean diff: {mean_diff:.6f}")
        
        if max_diff < 1e-3:
            logger.info("✅ cuBLASLt result matches theoretical expectation")
        else:
            logger.warning("⚠️ cuBLASLt result differs from theoretical expectation")
            
    except Exception as e:
        logger.error(f"cuBLASLt failed: {e}")

if __name__ == "__main__":
    test_theoretical_result()

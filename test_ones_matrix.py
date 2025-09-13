#!/usr/bin/env python3
"""
使用全为1的矩阵测试FP4 GEMM，便于发现数值问题
"""

import torch
import logging
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.custom_ops import nvfp4_gemm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_ones_matrix():
    """使用全为1的矩阵测试两个后端"""
    logger.info("=" * 60)
    logger.info("Testing with all-ones matrices")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available, skipping test")
        return False
    
    # 设置测试参数
    m, n, k = 64, 128, 256
    k_compressed = k // 2
    scale_groups = k_compressed // 16
    
    logger.info(f"Matrix dimensions: m={m}, n={n}, k={k}")
    logger.info(f"Compressed k: {k_compressed}, scale_groups: {scale_groups}")
    
    # 创建全为1的测试数据
    act_fp4 = torch.ones((m, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device="cuda")
    weight = torch.ones((n, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device="cuda")
    
    # 统一的scaling factor类型 (uint8)
    act_sf = torch.ones((m, scale_groups), dtype=torch.uint8, device="cuda")
    weight_scale = torch.ones((n, scale_groups), dtype=torch.uint8, device="cuda")
    alpha = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    
    logger.info(f"Input shapes: act_fp4={list(act_fp4.shape)}, weight={list(weight.shape)}")
    logger.info(f"Scale shapes: act_sf={list(act_sf.shape)}, weight_scale={list(weight_scale.shape)}")
    logger.info(f"Scale dtypes: act_sf={act_sf.dtype}, weight_scale={weight_scale.dtype}")
    
    # 打印一些输入值用于验证
    logger.info(f"act_fp4 sample values: {act_fp4[0, :5]}")
    logger.info(f"weight sample values: {weight[0, :5]}")
    logger.info(f"act_sf sample values: {act_sf[0, :5]}")
    logger.info(f"weight_scale sample values: {weight_scale[0, :5]}")
    
    try:
        # 测试CUTLASS后端
        logger.info("\n" + "="*40)
        logger.info("Testing CUTLASS backend...")
        logger.info("="*40)
        
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
        logger.info(f"CUTLASS result dtype: {result_cutlass.dtype}")
        logger.info(f"CUTLASS result range: [{torch.min(result_cutlass):.6f}, {torch.max(result_cutlass):.6f}]")
        logger.info(f"CUTLASS result mean: {torch.mean(result_cutlass):.6f}")
        logger.info(f"CUTLASS result std: {torch.std(result_cutlass):.6f}")
        logger.info(f"CUTLASS sample values: {result_cutlass[0, :5]}")
        
        # 检查CUTLASS结果
        cutlass_has_nan = torch.isnan(result_cutlass).any()
        cutlass_has_inf = torch.isinf(result_cutlass).any()
        logger.info(f"CUTLASS has NaN: {cutlass_has_nan}, has Inf: {cutlass_has_inf}")
        
        # 测试cuBLASLt后端
        logger.info("\n" + "="*40)
        logger.info("Testing cuBLASLt backend...")
        logger.info("="*40)
        
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
        logger.info(f"cuBLASLt result dtype: {result_cublaslt.dtype}")
        logger.info(f"cuBLASLt result range: [{torch.min(result_cublaslt):.6f}, {torch.max(result_cublaslt):.6f}]")
        logger.info(f"cuBLASLt result mean: {torch.mean(result_cublaslt):.6f}")
        logger.info(f"cuBLASLt result std: {torch.std(result_cublaslt):.6f}")
        logger.info(f"cuBLASLt sample values: {result_cublaslt[0, :5]}")
        
        # 检查cuBLASLt结果
        cublaslt_has_nan = torch.isnan(result_cublaslt).any()
        cublaslt_has_inf = torch.isinf(result_cublaslt).any()
        logger.info(f"cuBLASLt has NaN: {cublaslt_has_nan}, has Inf: {cublaslt_has_inf}")
        
        # 比较结果
        logger.info("\n" + "="*40)
        logger.info("Comparing results...")
        logger.info("="*40)
        
        abs_diff = torch.abs(result_cutlass - result_cublaslt)
        rel_diff = abs_diff / (torch.abs(result_cutlass) + 1e-8)
        
        max_abs_diff = torch.max(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()
        
        logger.info(f"Max absolute difference: {max_abs_diff:.6f}")
        logger.info(f"Max relative difference: {max_rel_diff:.6f}")
        logger.info(f"Mean absolute difference: {mean_abs_diff:.6f}")
        logger.info(f"Mean relative difference: {mean_rel_diff:.6f}")
        
        # 检查差异分布
        logger.info(f"Difference range: [{torch.min(abs_diff):.6f}, {torch.max(abs_diff):.6f}]")
        logger.info(f"Difference std: {torch.std(abs_diff):.6f}")
        
        # 检查是否在合理范围内
        tolerance = 1e-3
        if max_abs_diff < tolerance:
            logger.info("✅ Results are within tolerance - test PASSED")
            return True
        else:
            logger.warning(f"⚠️ Results differ by {max_abs_diff:.6f} (tolerance: {tolerance})")
            
            # 找出差异最大的位置
            max_diff_idx = torch.argmax(abs_diff)
            max_diff_pos = torch.unravel_index(max_diff_idx, abs_diff.shape)
            logger.info(f"Max difference at position: {max_diff_pos}")
            logger.info(f"CUTLASS value at max diff: {result_cutlass[max_diff_pos]:.6f}")
            logger.info(f"cuBLASLt value at max diff: {result_cublaslt[max_diff_pos]:.6f}")
            
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ones_matrix()
    exit(0 if success else 1)

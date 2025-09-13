#!/usr/bin/env python3
"""
测试scaling factor类型一致性
验证CUTLASS和cuBLASLt现在使用相同的scaling factor类型
"""

import torch
import logging
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.custom_ops import nvfp4_gemm

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_scaling_factor_consistency():
    """测试scaling factor类型一致性"""
    logger.info("=" * 60)
    logger.info("Testing scaling factor type consistency")
    logger.info("=" * 60)
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available, skipping test")
        return False
    
    # 设置测试参数
    m, n, k = 64, 128, 256
    k_compressed = k // 2
    scale_groups = k_compressed // 16
    
    # 创建测试数据 - 现在两个后端使用相同的scaling factor类型
    act_fp4 = torch.randint(0, 255, (m, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device="cuda")
    weight = torch.randint(0, 255, (n, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device="cuda")
    
    # 统一的scaling factor类型 (uint8)
    act_sf = torch.ones((m, scale_groups), dtype=torch.uint8, device="cuda")
    weight_scale = torch.ones((n, scale_groups), dtype=torch.uint8, device="cuda")
    alpha = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    
    logger.info(f"Input shapes: act_fp4={list(act_fp4.shape)}, weight={list(weight.shape)}")
    logger.info(f"Scale shapes: act_sf={list(act_sf.shape)}, weight_scale={list(weight_scale.shape)}")
    logger.info(f"Scale dtypes: act_sf={act_sf.dtype}, weight_scale={weight_scale.dtype}")
    
    try:
        # 测试CUTLASS后端
        logger.info("\nTesting CUTLASS backend...")
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
        logger.info(f"CUTLASS result: shape={list(result_cutlass.shape)}, dtype={result_cutlass.dtype}")
        
        # 测试cuBLASLt后端
        logger.info("\nTesting cuBLASLt backend...")
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
        logger.info(f"cuBLASLt result: shape={list(result_cublaslt.shape)}, dtype={result_cublaslt.dtype}")
        
        # 比较结果
        logger.info("\nComparing results...")
        abs_diff = torch.abs(result_cutlass - result_cublaslt)
        max_abs_diff = torch.max(abs_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()
        
        logger.info(f"Max absolute difference: {max_abs_diff:.6f}")
        logger.info(f"Mean absolute difference: {mean_abs_diff:.6f}")
        
        # 检查是否在合理范围内
        tolerance = 1e-3
        if max_abs_diff < tolerance:
            logger.info("✅ Results are within tolerance - scaling factor consistency test PASSED")
            return True
        else:
            logger.warning(f"⚠️ Results differ by {max_abs_diff:.6f} (tolerance: {tolerance})")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_scaling_factor_consistency()
    exit(0 if success else 1)

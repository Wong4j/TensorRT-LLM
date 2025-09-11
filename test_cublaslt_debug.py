#!/usr/bin/env python3
"""
调试 cuBLASLt FP4 GEMM 的简单测试脚本
"""

import logging
import torch
import traceback

# 设置详细日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_cublaslt_debug():
    """调试 cuBLASLt 后端"""
    logger.info("=" * 60)
    logger.info("cuBLASLt FP4 GEMM 调试测试")
    logger.info("=" * 60)
    
    try:
        from tensorrt_llm._torch.custom_ops import nvfp4_gemm
        import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
        
        # 创建简单的测试数据
        m, n, k = 32, 64, 128
        k_compressed = k // 2
        scale_groups = k_compressed // 16
        
        logger.info(f"测试矩阵大小: m={m}, n={n}, k={k}")
        logger.info(f"压缩后 K: {k_compressed}, 缩放因子组数: {scale_groups}")
        
        # 创建测试数据
        act_fp4 = torch.randint(0, 255, (m, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device="cuda")
        weight = torch.randint(0, 255, (n, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device="cuda")
        act_sf = torch.ones((m, scale_groups), dtype=torch.float8_e4m3fn, device="cuda")
        weight_scale = torch.ones((n, scale_groups), dtype=torch.float8_e4m3fn, device="cuda")
        alpha = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        
        logger.info("输入数据创建完成")
        logger.info(f"  act_fp4: {act_fp4.shape}, {act_fp4.dtype}")
        logger.info(f"  weight: {weight.shape}, {weight.dtype}")
        logger.info(f"  act_sf: {act_sf.shape}, {act_sf.dtype}")
        logger.info(f"  weight_scale: {weight_scale.shape}, {weight_scale.dtype}")
        
        # 测试 cuBLASLt 后端
        logger.info("开始测试 cuBLASLt 后端...")
        
        try:
            result = nvfp4_gemm(
                act_fp4=act_fp4,
                weight=weight,
                act_sf=act_sf,
                weight_scale=weight_scale,
                alpha=alpha,
                output_dtype=torch.bfloat16,
                to_userbuffers=False,
                backend="cublaslt"
            )
            
            logger.info("✅ cuBLASLt 后端测试成功！")
            logger.info(f"输出形状: {result.shape}")
            logger.info(f"输出类型: {result.dtype}")
            logger.info(f"输出设备: {result.device}")
            
        except Exception as e:
            logger.error(f"❌ cuBLASLt 后端测试失败: {e}")
            logger.error(f"异常类型: {type(e).__name__}")
            logger.error(f"详细错误信息:")
            logger.error(traceback.format_exc())
            
            # 尝试获取更详细的 CUDA 错误信息
            if hasattr(e, 'cudaError'):
                logger.error(f"CUDA 错误代码: {e.cudaError}")
            
            return False
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ 导入错误: {e}")
        logger.error(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"❌ 意外错误: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = test_cublaslt_debug()
    if success:
        print("\n🎉 测试成功！")
    else:
        print("\n💥 测试失败！")
    exit(0 if success else 1)

#!/usr/bin/env python3
"""
测试 cuBLASLt FP4 GEMM 是否启用
"""

import torch
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cublaslt_fp4_gemm_enabled():
    """测试 cuBLASLt FP4 GEMM 是否启用"""
    logger.info("Testing cuBLASLt FP4 GEMM availability...")
    
    try:
        # 检查是否有 cublaslt_nvfp4_gemm 操作
        if hasattr(torch.ops.trtllm, 'cublaslt_nvfp4_gemm'):
            logger.info("✅ cuBLASLt FP4 GEMM operation is available")
            return True
        else:
            logger.error("❌ cuBLASLt FP4 GEMM operation is NOT available")
            return False
    except Exception as e:
        logger.error(f"❌ Error checking cuBLASLt FP4 GEMM: {e}")
        return False

def test_compile_time_flags():
    """测试编译时标志"""
    logger.info("Checking compile-time flags...")
    
    try:
        # 尝试导入相关模块
        import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
        logger.info("✅ FP4 utils module available")
        
        # 检查数据类型定义
        logger.info(f"FLOAT4_E2M1X2 dtype: {fp4_utils.FLOAT4_E2M1X2}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error importing FP4 utils: {e}")
        return False

def test_cublaslt_backend():
    """测试 cuBLASLt 后端是否可用"""
    logger.info("Testing cuBLASLt backend...")
    
    try:
        from tensorrt_llm._torch.custom_ops import nvfp4_gemm
        import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
        
        # 创建测试数据
        m, n, k = 64, 128, 256
        k_compressed = k // 2  # FP4 压缩了 K 维度
        
        act_fp4 = torch.randint(0, 255, (m, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device="cuda")
        weight = torch.randint(0, 255, (n, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device="cuda")
        # nvfp4: 每 16 个元素共享一个缩放因子
        act_sf = torch.ones((m, k_compressed // 16), dtype=torch.float32, device="cuda")
        weight_scale = torch.ones((n, k_compressed // 16), dtype=torch.float32, device="cuda")
        alpha = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        
        logger.info("Testing cuBLASLt backend...")
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
            logger.info("✅ cuBLASLt backend test successful")
            logger.info(f"Output shape: {list(result.shape)}, dtype: {result.dtype}")
            return True
        except Exception as e:
            logger.error(f"❌ cuBLASLt backend test failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing cuBLASLt backend: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("cuBLASLt FP4 GEMM 启用测试")
    logger.info("=" * 60)
    
    # 测试编译时标志
    compile_ok = test_compile_time_flags()
    
    # 测试操作可用性
    ops_ok = test_cublaslt_fp4_gemm_enabled()
    
    # 测试后端功能
    backend_ok = test_cublaslt_backend()
    
    logger.info("=" * 60)
    logger.info("测试结果总结:")
    logger.info(f"编译时标志: {'✅ 通过' if compile_ok else '❌ 失败'}")
    logger.info(f"操作可用性: {'✅ 通过' if ops_ok else '❌ 失败'}")
    logger.info(f"后端功能: {'✅ 通过' if backend_ok else '❌ 失败'}")
    
    if compile_ok and ops_ok and backend_ok:
        logger.info("🎉 cuBLASLt FP4 GEMM 已成功启用！")
        return True
    else:
        logger.error("💥 cuBLASLt FP4 GEMM 启用失败，请检查编译配置")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

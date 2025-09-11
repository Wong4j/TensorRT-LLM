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

def test_cutlass_backend():
    """测试 CUTLASS 后端是否可用"""
    logger.info("Testing CUTLASS backend...")
    
    try:
        from tensorrt_llm._torch.custom_ops import nvfp4_gemm
        import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
        
        # 创建测试数据
        m, n, k = 64, 128, 256
        k_compressed = k // 2  # FP4 压缩了 K 维度
        
        act_fp4 = torch.randint(0, 255, (m, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device="cuda")
        weight = torch.randint(0, 255, (n, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device="cuda")
        # nvfp4: 每 16 个元素共享一个缩放因子
        # CUTLASS 期望 uint8 类型 (cutlass::float_ue4m3_t)
        act_sf = torch.ones((m, k_compressed // 16), dtype=torch.uint8, device="cuda")
        weight_scale = torch.ones((n, k_compressed // 16), dtype=torch.uint8, device="cuda")
        alpha = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        
        logger.info("Testing CUTLASS backend...")
        try:
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
            logger.info("✅ CUTLASS backend test successful")
            logger.info(f"Output shape: {list(result.shape)}, dtype: {result.dtype}")
            return True
        except Exception as e:
            logger.error(f"❌ CUTLASS backend test failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing CUTLASS backend: {e}")
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
        # cuBLASLt 期望 fp8_e4m3fn 类型
        act_sf = torch.ones((m, k_compressed // 16), dtype=torch.float8_e4m3fn, device="cuda")
        weight_scale = torch.ones((n, k_compressed // 16), dtype=torch.float8_e4m3fn, device="cuda")
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
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error testing cuBLASLt backend: {e}")
        return False

def test_backend_comparison():
    """对比测试 CUTLASS 和 cuBLASLt 两个后端"""
    logger.info("Comparing CUTLASS and cuBLASLt backends...")
    
    try:
        from tensorrt_llm._torch.custom_ops import nvfp4_gemm
        import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
        import time
        
        # 创建测试数据
        m, n, k = 128, 256, 512
        k_compressed = k // 2  # FP4 压缩了 K 维度
        
        act_fp4 = torch.randint(0, 255, (m, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device="cuda")
        weight = torch.randint(0, 255, (n, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device="cuda")
        # 为不同后端创建不同数据类型的缩放因子
        act_sf_uint8 = torch.ones((m, k_compressed // 16), dtype=torch.uint8, device="cuda")  # CUTLASS
        weight_scale_uint8 = torch.ones((n, k_compressed // 16), dtype=torch.uint8, device="cuda")  # CUTLASS
        act_sf_fp8 = torch.ones((m, k_compressed // 16), dtype=torch.float8_e4m3fn, device="cuda")  # cuBLASLt
        weight_scale_fp8 = torch.ones((n, k_compressed // 16), dtype=torch.float8_e4m3fn, device="cuda")  # cuBLASLt
        alpha = torch.tensor(1.0, dtype=torch.float32, device="cuda")
        
        results = {}
        
        # 测试 CUTLASS 后端
        logger.info("Testing CUTLASS backend...")
        try:
            start_time = time.time()
            result_cutlass = nvfp4_gemm(
                act_fp4=act_fp4,
                weight=weight,
                act_sf=act_sf_uint8,  # CUTLASS 使用 uint8
                weight_scale=weight_scale_uint8,  # CUTLASS 使用 uint8
                alpha=alpha,
                output_dtype=torch.bfloat16,
                to_userbuffers=False,
                backend="cutlass"
            )
            cutlass_time = time.time() - start_time
            results['cutlass'] = {
                'success': True,
                'shape': list(result_cutlass.shape),
                'dtype': result_cutlass.dtype,
                'time': cutlass_time
            }
            logger.info(f"✅ CUTLASS: shape={list(result_cutlass.shape)}, dtype={result_cutlass.dtype}, time={cutlass_time:.4f}s")
        except Exception as e:
            results['cutlass'] = {'success': False, 'error': str(e)}
            logger.error(f"❌ CUTLASS failed: {e}")
        
        # 测试 cuBLASLt 后端
        logger.info("Testing cuBLASLt backend...")
        try:
            start_time = time.time()
            result_cublaslt = nvfp4_gemm(
                act_fp4=act_fp4,
                weight=weight,
                act_sf=act_sf_fp8,  # cuBLASLt 使用 fp8_e4m3fn
                weight_scale=weight_scale_fp8,  # cuBLASLt 使用 fp8_e4m3fn
                alpha=alpha,
                output_dtype=torch.bfloat16,
                to_userbuffers=False,
                backend="cublaslt"
            )
            cublaslt_time = time.time() - start_time
            results['cublaslt'] = {
                'success': True,
                'shape': list(result_cublaslt.shape),
                'dtype': result_cublaslt.dtype,
                'time': cublaslt_time
            }
            logger.info(f"✅ cuBLASLt: shape={list(result_cublaslt.shape)}, dtype={result_cublaslt.dtype}, time={cublaslt_time:.4f}s")
        except Exception as e:
            results['cublaslt'] = {'success': False, 'error': str(e)}
            logger.error(f"❌ cuBLASLt failed: {e}")
        
        # 对比结果
        logger.info("=" * 50)
        logger.info("Backend Comparison Results:")
        logger.info("=" * 50)
        
        for backend, result in results.items():
            if result['success']:
                logger.info(f"{backend.upper()}: ✅ Success")
                logger.info(f"  Shape: {result['shape']}")
                logger.info(f"  Dtype: {result['dtype']}")
                logger.info(f"  Time:  {result['time']:.4f}s")
            else:
                logger.info(f"{backend.upper()}: ❌ Failed - {result['error']}")
        
        # 如果两个后端都成功，比较性能
        if results['cutlass']['success'] and results['cublaslt']['success']:
            cutlass_time = results['cutlass']['time']
            cublaslt_time = results['cublaslt']['time']
            speedup = cutlass_time / cublaslt_time if cublaslt_time > 0 else float('inf')
            logger.info(f"Performance comparison:")
            logger.info(f"  CUTLASS time:    {cutlass_time:.4f}s")
            logger.info(f"  cuBLASLt time:   {cublaslt_time:.4f}s")
            logger.info(f"  Speedup ratio:   {speedup:.2f}x")
        
        return results['cutlass']['success'] and results['cublaslt']['success']
        
    except Exception as e:
        logger.error(f"❌ Error in backend comparison: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("=" * 60)
    logger.info("FP4 GEMM 后端对比测试")
    logger.info("=" * 60)
    
    # 测试编译时标志
    compile_ok = test_compile_time_flags()
    
    # 测试操作可用性
    ops_ok = test_cublaslt_fp4_gemm_enabled()
    
    # 测试 CUTLASS 后端
    cutlass_ok = test_cutlass_backend()
    
    # 测试 cuBLASLt 后端
    cublaslt_ok = test_cublaslt_backend()
    
    # 对比测试两个后端
    comparison_ok = test_backend_comparison()
    
    logger.info("=" * 60)
    logger.info("测试结果总结:")
    logger.info(f"编译时标志: {'✅ 通过' if compile_ok else '❌ 失败'}")
    logger.info(f"操作可用性: {'✅ 通过' if ops_ok else '❌ 失败'}")
    logger.info(f"CUTLASS 后端: {'✅ 通过' if cutlass_ok else '❌ 失败'}")
    logger.info(f"cuBLASLt 后端: {'✅ 通过' if cublaslt_ok else '❌ 失败'}")
    logger.info(f"后端对比: {'✅ 通过' if comparison_ok else '❌ 失败'}")
    
    if compile_ok and ops_ok and cutlass_ok and cublaslt_ok and comparison_ok:
        logger.info("🎉 所有测试通过！CUTLASS 和 cuBLASLt 两个后端都可用！")
        return True
    else:
        logger.error("💥 部分测试失败，请检查配置")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

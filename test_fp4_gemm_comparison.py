#!/usr/bin/env python3
"""
FP4 GEMM 后端对比测试脚本
对比 CUTLASS 和 cuBLASLt 两个后端的性能和正确性
"""

import logging
import torch
import time
import numpy as np
from typing import Dict, List, Tuple

# 设置日志级别
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data(m: int, n: int, k: int, device: str = "cuda") -> Dict[str, torch.Tensor]:
    """创建测试数据"""
    k_compressed = k // 2  # FP4 压缩了 k 维度
    scale_groups = k_compressed // 16  # 每 16 个元素共享一个缩放因子
    
    # 导入 FP4 工具
    import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
    
    data = {
        'act_fp4': torch.randint(0, 255, (m, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device=device),
        'weight': torch.randint(0, 255, (n, k_compressed), dtype=fp4_utils.FLOAT4_E2M1X2, device=device),
        # 为不同后端创建不同数据类型的缩放因子
        'act_sf_uint8': torch.ones((m, scale_groups), dtype=torch.uint8, device=device),  # CUTLASS
        'weight_scale_uint8': torch.ones((n, scale_groups), dtype=torch.uint8, device=device),  # CUTLASS
        'act_sf_fp8': torch.ones((m, scale_groups), dtype=torch.float8_e4m3fn, device=device),  # cuBLASLt
        'weight_scale_fp8': torch.ones((n, scale_groups), dtype=torch.float8_e4m3fn, device=device),  # cuBLASLt
        'alpha': torch.tensor(1.0, dtype=torch.float32, device=device)
    }
    
    return data

def benchmark_backend(backend: str, data: Dict[str, torch.Tensor], 
                     output_dtype: torch.dtype, num_runs: int = 10) -> Dict:
    """基准测试单个后端"""
    from tensorrt_llm._torch.custom_ops import nvfp4_gemm
    
    logger.info(f"Benchmarking {backend.upper()} backend...")
    
    # 根据后端选择正确的缩放因子类型
    if backend == "cutlass":
        act_sf = data['act_sf_uint8']
        weight_scale = data['weight_scale_uint8']
    else:  # cublaslt
        act_sf = data['act_sf_fp8']
        weight_scale = data['weight_scale_fp8']
    
    # 预热
    try:
        _ = nvfp4_gemm(
            act_fp4=data['act_fp4'],
            weight=data['weight'],
            act_sf=act_sf,
            weight_scale=weight_scale,
            alpha=data['alpha'],
            output_dtype=output_dtype,
            to_userbuffers=False,
            backend=backend
        )
    except Exception as e:
        logger.error(f"Error in {backend} backend warmup: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': str(e)}
    
    # 基准测试
    times = []
    results = []
    
    for i in range(num_runs):
        torch.cuda.synchronize()  # 确保 GPU 操作完成
        start_time = time.time()
        
        try:
            result = nvfp4_gemm(
                act_fp4=data['act_fp4'],
                weight=data['weight'],
                act_sf=act_sf,
                weight_scale=weight_scale,
                alpha=data['alpha'],
                output_dtype=output_dtype,
                to_userbuffers=False,
                backend=backend
            )
            
            torch.cuda.synchronize()  # 确保 GPU 操作完成
            end_time = time.time()
            
            times.append(end_time - start_time)
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error in {backend} backend: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}
    
    # 计算统计信息
    times = np.array(times)
    
    return {
        'success': True,
        'times': times,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'shape': list(results[0].shape),
        'dtype': results[0].dtype,
        'results': results
    }

def compare_results(result1: torch.Tensor, result2: torch.Tensor, 
                   tolerance: float = 1e-3) -> Dict:
    """比较两个结果张量的差异"""
    try:
        # 计算绝对误差和相对误差
        abs_diff = torch.abs(result1 - result2)
        rel_diff = abs_diff / (torch.abs(result1) + 1e-8)
        
        max_abs_diff = torch.max(abs_diff).item()
        max_rel_diff = torch.max(rel_diff).item()
        mean_abs_diff = torch.mean(abs_diff).item()
        mean_rel_diff = torch.mean(rel_diff).item()
        
        # 检查是否在容差范围内
        within_tolerance = max_abs_diff < tolerance
        
        return {
            'max_abs_diff': max_abs_diff,
            'max_rel_diff': max_rel_diff,
            'mean_abs_diff': mean_abs_diff,
            'mean_rel_diff': mean_rel_diff,
            'within_tolerance': within_tolerance
        }
    except Exception as e:
        return {'error': str(e)}

def test_different_sizes():
    """测试不同矩阵大小的性能"""
    logger.info("=" * 60)
    logger.info("Testing different matrix sizes")
    logger.info("=" * 60)
    
    test_sizes = [
        (64, 128, 256),
        (128, 256, 512),
        (256, 512, 1024),
        (512, 1024, 2048),
    ]
    
    results = {}
    
    for m, n, k in test_sizes:
        logger.info(f"\nTesting size: m={m}, n={n}, k={k}")
        
        # 创建测试数据
        data = create_test_data(m, n, k)
        
        # 测试两个后端
        cutlass_result = benchmark_backend("cutlass", data, torch.bfloat16, num_runs=5)
        cublaslt_result = benchmark_backend("cublaslt", data, torch.bfloat16, num_runs=5)
        
        results[(m, n, k)] = {
            'cutlass': cutlass_result,
            'cublaslt': cublaslt_result
        }
        
        # 打印结果
        if cutlass_result['success'] and cublaslt_result['success']:
            logger.info(f"CUTLASS:    {cutlass_result['mean_time']:.4f}s ± {cutlass_result['std_time']:.4f}s")
            logger.info(f"cuBLASLt:   {cublaslt_result['mean_time']:.4f}s ± {cublaslt_result['std_time']:.4f}s")
            
            speedup = cutlass_result['mean_time'] / cublaslt_result['mean_time']
            logger.info(f"Speedup:    {speedup:.2f}x")
            
            # 比较结果正确性
            comparison = compare_results(
                cutlass_result['results'][0], 
                cublaslt_result['results'][0]
            )
            if 'error' not in comparison:
                logger.info(f"Max abs diff: {comparison['max_abs_diff']:.6f}")
                logger.info(f"Max rel diff: {comparison['max_rel_diff']:.6f}")
                logger.info(f"Within tolerance: {comparison['within_tolerance']}")
        else:
            logger.error(f"CUTLASS success: {cutlass_result['success']}")
            logger.error(f"cuBLASLt success: {cublaslt_result['success']}")
    
    return results

def test_different_output_types():
    """测试不同输出数据类型"""
    logger.info("=" * 60)
    logger.info("Testing different output data types")
    logger.info("=" * 60)
    
    output_types = [torch.float16, torch.bfloat16, torch.float32]
    m, n, k = 128, 256, 512
    
    data = create_test_data(m, n, k)
    
    for output_dtype in output_types:
        logger.info(f"\nTesting output dtype: {output_dtype}")
        
        cutlass_result = benchmark_backend("cutlass", data, output_dtype, num_runs=3)
        cublaslt_result = benchmark_backend("cublaslt", data, output_dtype, num_runs=3)
        
        if cutlass_result['success'] and cublaslt_result['success']:
            logger.info(f"CUTLASS:    {cutlass_result['mean_time']:.4f}s")
            logger.info(f"cuBLASLt:   {cublaslt_result['mean_time']:.4f}s")
            logger.info(f"Output shape: {cutlass_result['shape']}")
            logger.info(f"Output dtype: {cutlass_result['dtype']}")
        else:
            logger.error(f"Failed for {output_dtype}")

def main():
    """主测试函数"""
    logger.info("=" * 80)
    logger.info("FP4 GEMM 后端对比测试")
    logger.info("=" * 80)
    
    if not torch.cuda.is_available():
        logger.error("CUDA not available, skipping test")
        return False
    
    try:
        # 测试不同矩阵大小
        size_results = test_different_sizes()
        
        # 测试不同输出类型
        test_different_output_types()
        
        # 总结
        logger.info("=" * 80)
        logger.info("测试完成！")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

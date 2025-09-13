#!/usr/bin/env python3

import torch
import tensorrt_llm
from tensorrt_llm.quantization.utils import fp4_utils
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_memory_layout():
    """测试uint8和float8_e4m3fn的内存布局是否相同"""
    
    # 创建测试数据
    test_values = [0x70, 0x78, 0x68, 0x60]  # 1.0, 2.0, 0.5, 0.25
    
    for val in test_values:
        logger.info(f"\n--- Testing value: 0x{val:02x} ---")
        
        # 方法1: 直接创建uint8 tensor
        uint8_tensor = torch.tensor([val] * 16, dtype=torch.uint8, device='cuda')
        logger.info(f"uint8_tensor: {uint8_tensor}")
        logger.info(f"uint8_tensor hex: {[hex(x) for x in uint8_tensor.cpu().numpy()]}")
        
        # 方法2: 创建float8_e4m3fn tensor
        try:
            # 先创建float8_e4m3fn tensor
            fp8_tensor = torch.tensor([val] * 16, dtype=torch.float8_e4m3fn, device='cuda')
            logger.info(f"fp8_tensor: {fp8_tensor}")
            
            # 检查内存布局
            uint8_bytes = uint8_tensor.cpu().numpy().tobytes()
            fp8_bytes = fp8_tensor.cpu().numpy().tobytes()
            
            logger.info(f"uint8 bytes: {uint8_bytes.hex()}")
            logger.info(f"fp8 bytes: {fp8_bytes.hex()}")
            logger.info(f"Memory layout identical: {uint8_bytes == fp8_bytes}")
            
            # 检查reinterpret_cast是否有效
            reinterpreted = uint8_tensor.view(torch.float8_e4m3fn)
            logger.info(f"reinterpreted: {reinterpreted}")
            logger.info(f"reinterpreted hex: {[hex(x) for x in reinterpreted.cpu().numpy()]}")
            
        except Exception as e:
            logger.error(f"Error creating fp8_tensor: {e}")
        
        # 方法3: 检查C++中的__nv_fp8_e4m3
        logger.info("Checking C++ __nv_fp8_e4m3 compatibility...")
        
        # 尝试直接使用uint8数据作为scaling factor
        try:
            # 模拟cuBLASLt调用
            act_fp4 = torch.ones((64, 128), dtype=fp4_utils.FLOAT4_E2M1X2, device='cuda')
            weight = torch.ones((128, 128), dtype=fp4_utils.FLOAT4_E2M1X2, device='cuda')
            
            # 使用uint8作为scaling factor
            act_sf_uint8 = torch.tensor([val] * 8, dtype=torch.uint8, device='cuda')
            weight_sf_uint8 = torch.tensor([val] * 8, dtype=torch.uint8, device='cuda')
            
            logger.info(f"act_sf_uint8: {act_sf_uint8}")
            logger.info(f"weight_sf_uint8: {weight_sf_uint8}")
            
            # 检查内存对齐
            logger.info(f"act_sf_uint8 data_ptr: {hex(act_sf_uint8.data_ptr())}")
            logger.info(f"act_sf_uint8 alignment: {act_sf_uint8.data_ptr() % 16}")
            
        except Exception as e:
            logger.error(f"Error in cuBLASLt simulation: {e}")

if __name__ == "__main__":
    test_memory_layout()

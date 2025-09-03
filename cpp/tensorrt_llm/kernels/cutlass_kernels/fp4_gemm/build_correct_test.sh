#!/bin/bash

echo "=== 正确的 cuBLASLt FP4 GEMM 测试 ==="

# 检查 CUDA 版本
echo "CUDA 版本: $(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//'),"
echo "GPU 信息:"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits

# 清理旧的构建目录
echo "清理旧的构建目录..."
rm -rf build_correct_test

# 创建构建目录
mkdir -p build_correct_test
cd build_correct_test

# 配置 CMake
echo "配置 CMake..."
cp ../CMakeLists_correct.txt CMakeLists.txt
cmake . -DCMAKE_BUILD_TYPE=Release

# 编译
echo "开始编译..."
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "编译完成！"
    echo "✓ 可执行文件创建成功"
    echo ""
    echo "=== 运行正确的 FP4 GEMM 测试 ==="
    ./correct_fp4_gemm_test
else
    echo "编译失败！"
    exit 1
fi

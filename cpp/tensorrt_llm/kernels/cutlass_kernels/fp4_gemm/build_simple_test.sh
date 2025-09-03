#!/bin/bash

# 简化 cuBLASLt FP4 GEMM 测试构建和运行脚本

set -e

echo "=== 简化 cuBLASLt FP4 GEMM 功能测试 ==="

# 检查 CUDA 环境
if ! command -v nvcc &> /dev/null; then
    echo "错误: 未找到 nvcc，请确保 CUDA 环境已正确安装"
    exit 1
fi

echo "CUDA 版本: $(nvcc --version | grep release | cut -d' ' -f5)"

# 检查 GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未找到 nvidia-smi，请确保 NVIDIA 驱动已安装"
    exit 1
fi

echo "GPU 信息:"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader,nounits

# 创建构建目录
BUILD_DIR="build_simple_test"
if [ -d "$BUILD_DIR" ]; then
    echo "清理旧的构建目录..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "配置 CMake..."
# 复制自定义 CMakeLists.txt 到构建目录
cp ../CMakeLists_simple.txt ./CMakeLists.txt

cmake . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"

echo "开始编译..."
make -j$(nproc)

echo "编译完成！"

# 检查可执行文件
if [ -f "simple_test_cublaslt_fp4_gemm" ]; then
    echo "✓ 可执行文件创建成功"
else
    echo "✗ 可执行文件创建失败"
    exit 1
fi

echo ""
echo "=== 运行测试 ==="
echo ""

# 运行测试
./simple_test_cublaslt_fp4_gemm

echo ""
echo "=== 测试完成 ==="

# 返回上级目录
cd ..

echo "构建目录: $BUILD_DIR"
echo "可执行文件: $BUILD_DIR/simple_test_cublaslt_fp4_gemm"
echo ""
echo "如需重新运行测试，请执行:"
echo "cd $BUILD_DIR && ./simple_test_cublaslt_fp4_gemm"

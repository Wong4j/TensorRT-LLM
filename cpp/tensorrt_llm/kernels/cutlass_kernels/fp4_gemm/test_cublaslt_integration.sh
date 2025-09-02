#!/bin/bash

# 测试 cuBLASLt FP4 GEMM 集成脚本
# 用法: ./test_cublaslt_integration.sh [--enable-cublaslt]

set -e

# 默认参数
ENABLE_CUBLASLT=OFF
BUILD_DIR="build_test_cublaslt"
CLEAN_BUILD=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --enable-cublaslt)
            ENABLE_CUBLASLT=ON
            shift
            ;;
        --clean)
            CLEAN_BUILD=true
            shift
            ;;
        --help)
            echo "用法: $0 [选项]"
            echo "选项:"
            echo "  --enable-cublaslt    启用 cuBLASLt 后端"
            echo "  --clean              清理构建目录"
            echo "  --help               显示此帮助信息"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "测试 cuBLASLt FP4 GEMM 集成"
echo "=========================================="
echo "启用 cuBLASLt: $ENABLE_CUBLASLT"
echo "构建目录: $BUILD_DIR"
echo ""

# 检查是否在正确的目录
if [ ! -f "CMakeLists.txt" ]; then
    echo "错误: 请在 TensorRT-LLM 根目录运行此脚本"
    exit 1
fi

# 清理构建目录
if [ "$CLEAN_BUILD" = true ]; then
    echo "[INFO] 清理构建目录..."
    rm -rf $BUILD_DIR
fi

# 创建构建目录
if [ ! -d "$BUILD_DIR" ]; then
    echo "[INFO] 创建构建目录: $BUILD_DIR"
    mkdir -p $BUILD_DIR
fi

cd $BUILD_DIR

# 配置 CMake
echo "[INFO] 配置 CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSING_OSS_CUTLASS_FP4_GEMM=ON \
    -DENABLE_CUBLASLT_FP4=$ENABLE_CUBLASLT \
    -DCMAKE_CUDA_ARCHITECTURES="100;120" \
    -DENABLE_BF16=ON \
    -DBUILD_UNIT_TESTS=ON

# 构建
echo "[INFO] 开始构建..."
make -j$(nproc) fp4_gemm_src

echo ""
echo "=========================================="
echo "构建完成！"
echo "=========================================="

# 检查构建结果
if [ -f "cpp/tensorrt_llm/kernels/cutlass_kernels/libfp4_gemm_src.a" ]; then
    echo "✅ FP4 GEMM 库构建成功"
    
    # 检查是否包含 cuBLASLt 符号
    if [ "$ENABLE_CUBLASLT" = "ON" ]; then
        echo "[INFO] 检查 cuBLASLt 符号..."
        if nm libfp4_gemm_src.a | grep -q "executeCublasLtNvfp4Gemm"; then
            echo "✅ cuBLASLt 符号已包含在库中"
        else
            echo "⚠️  警告: 未找到 cuBLASLt 符号"
        fi
    fi
else
    echo "❌ FP4 GEMM 库构建失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "使用方法:"
echo "1. 编译时启用 cuBLASLt:"
echo "   cmake .. -DENABLE_CUBLASLT_FP4=ON"
echo ""
echo "2. 运行时启用 cuBLASLt:"
echo "   export TRTLLM_USE_CUBLASLT_FP4=1"
echo ""
echo "3. 检查日志中的后端选择信息:"
echo "   - 'Using cuBLASLt backend for NVFP4 GEMM' (cuBLASLt)"
echo "   - 'Using CUTLASS backend for FP4 GEMM' (CUTLASS)"

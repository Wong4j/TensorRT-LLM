# FP4 GEMM 架构设计文档

## 概述

本文档描述了 TensorRT-LLM 中 FP4 GEMM 的完整架构设计，包括多后端支持、命名空间组织、使用方法和最佳实践。

## 架构设计

### 设计原则

1. **单一职责原则**: 每个类只有一个改变的理由
2. **开闭原则**: 对扩展开放，对修改关闭
3. **依赖倒置原则**: 依赖抽象而不是具体实现
4. **接口隔离原则**: 客户端不依赖不需要的接口
5. **命名空间清晰**: 每个命名空间职责单一

### 命名空间结构

```
tensorrt_llm::kernels::
├── fp4_gemm::                    ← 统一接口和工厂
│   ├── IFp4GemmRunner           ← 统一接口
│   ├── Fp4GemmBackendFactory    ← 后端工厂
│   ├── FP4GemmType              ← GEMM 类型枚举
│   └── FP4GemmBackend           ← 后端类型枚举
├── cutlass_kernels::             ← 纯 CUTLASS 相关
│   └── CutlassFp4GemmRunner     ← CUTLASS 实现
└── cublaslt_kernels::            ← 纯 cuBLASLt 相关
    └── CublasLtFp4GemmRunner    ← cuBLASLt 实现
```

## 核心组件

### 1. 统一接口 (`fp4_gemm` 命名空间)

**文件**: `include/fp4_gemm.h`

```cpp
namespace tensorrt_llm::kernels::fp4_gemm {
    // GEMM 类型枚举
    enum class FP4GemmType {
        W4A4_NVFP4_NVFP4,    // 4-bit weights, 4-bit activations
        W4A8_MXFP4_MXFP8,    // 4-bit weights, 8-bit activations
    };

    // 后端类型枚举
    enum class FP4GemmBackend {
        CUTLASS,             // CUTLASS 后端
        CUBLASLT,            // cuBLASLt 后端
    };

    // 统一的 FP4 GEMM 接口
    class IFp4GemmRunner {
    public:
        virtual ~IFp4GemmRunner() = default;
        
        // 执行 FP4 GEMM 操作
        virtual void gemm(void* D, void const* A, void const* B, 
                         void const* input_sf, void const* weight_sf,
                         float const* global_sf, int m, int n, int k, 
                         int batch_count, tkc::CutlassGemmConfig gemmConfig,
                         char* workspace, const size_t workspaceBytes, 
                         cudaStream_t stream) = 0;

        // 获取工作空间大小
        virtual size_t getWorkspaceSize(int const m, int const n, 
                                       int const k, int batch_count) = 0;

        // 获取支持的配置
        virtual std::vector<tkc::CutlassGemmConfig> getConfigs() const = 0;

        // 获取后端类型
        virtual FP4GemmBackend getBackendType() const = 0;

        // 检查是否支持指定的 GEMM 类型
        virtual bool supportsGemmType(FP4GemmType gemmType) const = 0;
    };

    // 后端工厂类
    class Fp4GemmBackendFactory {
    public:
        // 创建 FP4 GEMM Runner
        template <typename T>
        static std::unique_ptr<IFp4GemmRunner> createRunner(
            FP4GemmType gemmType, 
            FP4GemmBackend backend = FP4GemmBackend::CUTLASS);

        // 获取可用的后端列表
        static std::vector<FP4GemmBackend> getAvailableBackends();

        // 检查后端是否可用
        static bool isBackendAvailable(FP4GemmBackend backend);

        // 获取推荐的后端
        static FP4GemmBackend getRecommendedBackend(FP4GemmType gemmType);
    };

    // 向后兼容
    using CutlassFp4GemmRunnerInterface = IFp4GemmRunner;
}
```

### 2. CUTLASS 后端 (`cutlass_kernels` 命名空间)

**文件**: `include/fp4_gemm.h` (在 `cutlass_kernels` 命名空间内)

```cpp
namespace tensorrt_llm::kernels::cutlass_kernels {
    template <typename T, fp4_gemm::FP4GemmType gemmType>
    class CutlassFp4GemmRunner : public virtual fp4_gemm::IFp4GemmRunner {
    public:
        CutlassFp4GemmRunner();
        ~CutlassFp4GemmRunner() override;

        // 实现 IFp4GemmRunner 接口
        void gemm(...) override;
        size_t getWorkspaceSize(...) override;
        std::vector<tkc::CutlassGemmConfig> getConfigs() const override;
        fp4_gemm::FP4GemmBackend getBackendType() const override;
        bool supportsGemmType(fp4_gemm::FP4GemmType gemmType) const override;

    private:
        // CUTLASS 特定的实现
        size_t dispatchToArch(...);
        size_t getWorkspaceSizeImpl(...);
        int mSm;
        int mMultiProcessorCount;
    };
}
```

### 3. cuBLASLt 后端 (`cublaslt_kernels` 命名空间)

**文件**: `include/fp4_gemm.h` (在 `cublaslt_kernels` 命名空间内)

```cpp
namespace tensorrt_llm::kernels::cublaslt_kernels {
    template <typename T, fp4_gemm::FP4GemmType gemmType>
    class CublasLtFp4GemmRunner : public fp4_gemm::IFp4GemmRunner {
    public:
        CublasLtFp4GemmRunner();
        ~CublasLtFp4GemmRunner() override;

        // 实现 IFp4GemmRunner 接口
        void gemm(...) override;
        size_t getWorkspaceSize(...) override;
        std::vector<tkc::CutlassGemmConfig> getConfigs() const override;
        fp4_gemm::FP4GemmBackend getBackendType() const override;
        bool supportsGemmType(fp4_gemm::FP4GemmType gemmType) const override;

    private:
        // cuBLASLt 特定的实现
        void executeCublasLtGemm(...);
    };
}
```

## 支持的功能

### 功能矩阵

| 功能 | CUTLASS 后端 | cuBLASLt 后端 |
|------|-------------|---------------|
| W4A4_NVFP4_NVFP4 | ✅ | ✅ |
| W4A8_MXFP4_MXFP8 | ✅ | ❌ |
| FP16 输出 | ✅ | ✅ |
| BF16 输出 | ✅ | ✅ |
| FP32 输出 | ✅ | ✅ |
| SM100 架构 | ✅ | ✅ |
| SM120 架构 | ✅ | ✅ |
| 自动调优 | ✅ | ✅ |
| 工作空间管理 | ✅ | ✅ |

### 性能特性

**CUTLASS 后端优势:**
- 完整功能支持
- 高度可定制
- 支持所有 GEMM 类型
- 更好的控制粒度

**cuBLASLt 后端优势:**
- 优化的库实现
- 自动算法选择
- 更好的内存管理
- 减少代码复杂度

## 编译配置

### 启用 cuBLASLt 支持

```bash
cmake .. \
    -DUSING_OSS_CUTLASS_FP4_GEMM=ON \
    -DENABLE_CUBLASLT_FP4=ON \
    -DCMAKE_CUDA_ARCHITECTURES="100;120" \
    -DENABLE_BF16=ON
```

### 编译选项说明

- `USING_OSS_CUTLASS_FP4_GEMM=ON`: 启用 FP4 GEMM 支持
- `ENABLE_CUBLASLT_FP4=ON`: 启用 cuBLASLt 后端支持
- `CMAKE_CUDA_ARCHITECTURES`: 指定目标 GPU 架构 (SM100, SM120)
- `ENABLE_BF16=ON`: 启用 BF16 支持

## 使用方法

### 1. 使用工厂模式 (推荐)

```cpp
#include "tensorrt_llm/kernels/cutlass_kernels/include/fp4_gemm.h"

using namespace tensorrt_llm::kernels::fp4_gemm;

// 创建 CUTLASS 后端
auto cutlass_runner = Fp4GemmBackendFactory::createRunner<half>(
    FP4GemmType::W4A4_NVFP4_NVFP4, 
    FP4GemmBackend::CUTLASS);

// 创建 cuBLASLt 后端
auto cublaslt_runner = Fp4GemmBackendFactory::createRunner<half>(
    FP4GemmType::W4A4_NVFP4_NVFP4, 
    FP4GemmBackend::CUBLASLT);

// 使用推荐的后端
auto recommended_backend = Fp4GemmBackendFactory::getRecommendedBackend(
    FP4GemmType::W4A4_NVFP4_NVFP4);
auto runner = Fp4GemmBackendFactory::createRunner<half>(
    FP4GemmType::W4A4_NVFP4_NVFP4, 
    recommended_backend);
```

### 2. 直接使用特定后端

```cpp
// 使用 CUTLASS 后端
#include "tensorrt_llm/kernels/cutlass_kernels/include/fp4_gemm.h"
using namespace tensorrt_llm::kernels::cutlass_kernels;
auto cutlass_runner = std::make_unique<CutlassFp4GemmRunner<half, 
    fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4>>();

// 使用 cuBLASLt 后端
#include "tensorrt_llm/kernels/cutlass_kernels/include/fp4_gemm.h"
using namespace tensorrt_llm::kernels::cublaslt_kernels;
auto cublaslt_runner = std::make_unique<CublasLtFp4GemmRunner<half, 
    fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4>>();
```

### 3. 向后兼容方式

```cpp
// 原有的方式仍然可以正常工作
#include "tensorrt_llm/kernels/cutlass_kernels/include/fp4_gemm.h"
using namespace tensorrt_llm::kernels::cutlass_kernels;
auto runner = std::make_unique<CutlassFp4GemmRunner<half, 
    fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4>>();
```

## 文件结构

```
fp4_gemm/
├── include/
│   └── fp4_gemm.h                    ← 统一接口 (包含所有后端声明)
├── fp4_gemm_template.h               ← CUTLASS 实现 (cutlass_kernels 命名空间)
├── cublaslt_fp4_gemm_runner.cu       ← cuBLASLt 实现 (cublaslt_kernels 命名空间)
├── fp4_gemm_backend_factory.cu       ← 工厂实现 (fp4_gemm 命名空间)
├── cublaslt_nvfp4_gemm.cu            ← cuBLASLt 底层实现
├── example_clean_architecture.cpp    ← 使用示例
└── FP4_GEMM_ARCHITECTURE.md          ← 本文档
```

### 统一头文件设计

所有 FP4 GEMM 相关的类声明都统一在 `include/fp4_gemm.h` 中，按命名空间组织：

```cpp
// fp4_gemm.h 文件结构
namespace tensorrt_llm::kernels::fp4_gemm {
    // 统一接口和工厂
    class IFp4GemmRunner { ... };
    class Fp4GemmBackendFactory { ... };
    enum class FP4GemmType { ... };
    enum class FP4GemmBackend { ... };
}

namespace tensorrt_llm::kernels::cutlass_kernels {
    // CUTLASS 后端
    template <typename T, fp4_gemm::FP4GemmType gemmType>
    class CutlassFp4GemmRunner : public fp4_gemm::IFp4GemmRunner { ... };
}

namespace tensorrt_llm::kernels::cublaslt_kernels {
    // cuBLASLt 后端
    template <typename T, fp4_gemm::FP4GemmType gemmType>
    class CublasLtFp4GemmRunner : public fp4_gemm::IFp4GemmRunner { ... };
}
```

**优势**:
- ✅ 单一头文件包含所有声明
- ✅ 减少包含依赖
- ✅ 简化使用方式
- ✅ 保持命名空间清晰

## 扩展性

### 添加新后端

```cpp
// 1. 实现接口
class NewBackendFp4GemmRunner : public fp4_gemm::IFp4GemmRunner {
    // 实现所有虚函数
};

// 2. 在工厂中注册
template <typename T>
std::unique_ptr<fp4_gemm::IFp4GemmRunner> Fp4GemmBackendFactory::createRunner(...) {
    switch (backend) {
    case fp4_gemm::FP4GemmBackend::NEW_BACKEND:
        return std::make_unique<NewBackendFp4GemmRunner<T, gemmType>>();
    }
}
```

### 添加新 GEMM 类型

```cpp
// 1. 扩展枚举
enum class FP4GemmType {
    W4A4_NVFP4_NVFP4,
    W4A8_MXFP4_MXFP8,
    NEW_GEMM_TYPE,  // 新类型
};

// 2. 在后端中实现支持
bool CutlassFp4GemmRunner::supportsGemmType(FP4GemmType gemmType) const {
    return gemmType == FP4GemmType::W4A4_NVFP4_NVFP4 ||
           gemmType == FP4GemmType::W4A8_MXFP4_MXFP8 ||
           gemmType == FP4GemmType::NEW_GEMM_TYPE;  // 新支持
}
```

## 测试策略

### 单元测试

```cpp
// 每个后端独立测试
TEST(CutlassFp4GemmRunner, BasicFunctionality) {
    auto runner = std::make_unique<CutlassFp4GemmRunner<half, 
        fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4>>();
    // 测试 CUTLASS 特定功能
}

TEST(CublasLtFp4GemmRunner, BasicFunctionality) {
    auto runner = std::make_unique<CublasLtFp4GemmRunner<half, 
        fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4>>();
    // 测试 cuBLASLt 特定功能
}
```

### 集成测试

```cpp
// 测试工厂模式
TEST(Fp4GemmBackendFactory, CreateRunner) {
    auto runner = Fp4GemmBackendFactory::createRunner<half>(
        fp4_gemm::FP4GemmType::W4A4_NVFP4_NVFP4, 
        fp4_gemm::FP4GemmBackend::CUTLASS);
    EXPECT_NE(runner, nullptr);
}
```

### 性能测试

```cpp
// 比较不同后端的性能
TEST(PerformanceComparison, CutlassVsCublasLt) {
    // 测试相同输入下的性能差异
}
```

## 调试和日志

### 日志信息

运行时会在日志中显示后端选择信息：

```
[TensorRT-LLM][INFO] Using cuBLASLt backend for NVFP4 GEMM
# 或
[TensorRT-LLM][DEBUG] Using CUTLASS backend for FP4 GEMM
```

### 调试技巧

1. **检查编译配置**:
   ```bash
   # 检查是否包含 cuBLASLt 符号
   nm libfp4_gemm_src.a | grep executeCublasLtNvfp4Gemm
   ```

2. **运行时调试**:
   ```bash
   # 启用详细日志
   export TLLM_LOG_LEVEL=DEBUG
   ```

3. **性能分析**:
   ```bash
   # 使用 nsys 分析性能
   nsys profile --trace=cuda ./your_program
   ```

## 故障排除

### 常见问题

1. **编译错误**: 确保安装了 cuBLASLt 库
2. **运行时错误**: 检查 GPU 架构是否支持 (SM100+)
3. **性能问题**: 尝试不同的后端或调整配置

### 错误信息

- `"cuBLASLt not enabled in this build"`: 编译时未启用 cuBLASLt 支持
- `"Unsupported output type for cuBLASLt NVFP4 GEMM"`: 不支持的输出类型
- `"No valid heuristic found for NVFP4 GEMM"`: cuBLASLt 无法找到合适的算法

## 最佳实践

### 1. 后端选择

- **W4A4 类型**: 优先使用 cuBLASLt (如果可用)
- **W4A8 类型**: 只能使用 CUTLASS
- **性能关键**: 根据具体用例进行性能测试

### 2. 内存管理

- 使用 RAII 模式管理资源
- 正确设置工作空间大小
- 及时释放 GPU 内存

### 3. 错误处理

- 使用 try-catch 块处理异常
- 检查返回值状态
- 提供有意义的错误信息

### 4. 性能优化

- 预热 GPU 内核
- 使用合适的 CUDA 流
- 避免频繁的内存分配

## 未来计划

### 短期计划
- [ ] 添加 `W4A8_MXFP4_MXFP8` 的 cuBLASLt 支持
- [ ] 优化 cuBLASLt 配置
- [ ] 添加性能基准测试

### 长期计划
- [ ] 自动后端选择算法
- [ ] 更多 GPU 架构支持
- [ ] 集成到 TensorRT-LLM 的 Python API

## 总结

这个 FP4 GEMM 架构设计提供了：

1. **清晰的职责分离**: 每个组件职责单一
2. **灵活的扩展性**: 易于添加新的后端和 GEMM 类型
3. **统一的接口**: 简化了使用方式
4. **向后兼容**: 现有代码无需修改
5. **良好的可维护性**: 代码组织清晰，易于理解和维护

这个设计符合现代 C++ 的最佳实践，为 TensorRT-LLM 的 FP4 GEMM 功能提供了坚实的基础。

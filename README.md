# MLXSharp

[![NuGet](https://img.shields.io/nuget/v/ManagedCode.MLXSharp.svg)](https://www.nuget.org/packages/ManagedCode.MLXSharp)
[![NuGet](https://img.shields.io/nuget/v/ManagedCode.MLXSharp.SemanticKernel.svg)](https://www.nuget.org/packages/ManagedCode.MLXSharp.SemanticKernel)

MLXSharp is the first .NET wrapper around [Apple MLX](https://github.com/ml-explore/mlx) that plugs straight into [`Microsoft.Extensions.AI`](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai). It is designed and tested on macOS (Apple silicon) but ships prebuilt native binaries for both macOS and Linux so that NuGet consumers can run out-of-the-box.

## Highlights
- .NET 9 / C# 13 preview friendly APIs that implement `IChatClient`, `IEmbeddingGenerator<string, Embedding<float>>`, and image helpers.
- Dependency Injection extensions (`AddMlx`) and Semantic Kernel integration (`AddMlxChatCompletion`).
- Native runtime loader that understands `MLXSHARP_LIBRARY`, custom paths, and packaged runtimes.
- CI pipeline that builds the managed code once, produces macOS and Linux native libraries in parallel, and packs everything into distributable NuGet packages.

## Quick Start
Add the package and wire it into the service collection:

```bash
dotnet add package ManagedCode.MLXSharp
```

```csharp
using Microsoft.Extensions.DependencyInjection;
using MLXSharp;

var services = new ServiceCollection();
services.AddMlx(builder =>
{
    builder.Configure(options =>
    {
        options.ChatModelId = "mlx-chat";
        options.EmbeddingModelId = "mlx-embedding";
    });

    builder.UseManagedBackend(new MlxManagedBackend());
    // Switch to the native backend when libmlxsharp.{dylib|so} is available.
    // builder.UseNativeBackend();
});

var provider = services.BuildServiceProvider();
var chat = provider.GetRequiredService<IChatClient>();
var reply = await chat.GetResponseAsync("hello MLX", CancellationToken.None);
```

### Semantic Kernel

```bash
dotnet add package ManagedCode.MLXSharp.SemanticKernel
```

```csharp
using Microsoft.SemanticKernel;
using MLXSharp.SemanticKernel;

var kernelBuilder = Kernel.CreateBuilder();
kernelBuilder.AddMlxChatCompletion(b => b.UseManagedBackend(new MlxManagedBackend()));
var kernel = kernelBuilder.Build();

var history = new ChatHistory();
history.AddUserMessage("Summarise MLX in one sentence");
var response = await kernel
    .GetRequiredService<IChatCompletionService>()
    .GetChatMessageContentsAsync(history, new PromptExecutionSettings(), kernel, CancellationToken.None);
```

## Repository Layout
```
extern/mlx/                  # MLX sources (git submodule)
native/                      # CMake project that builds libmlxsharp
src/MLXSharp/                # Managed MLXSharp library + runtime assets
src/MLXSharp.SemanticKernel/ # Semantic Kernel integration
src/MLXSharp.Tests/          # Integration tests
```

## Building the native wrapper locally
1. Install .NET 9, CMake, and ensure Xcode command-line tools (macOS) or build-essential + OpenBLAS/LAPACK headers (Linux) are available.
2. Sync submodules:
   ```bash
   git submodule update --init --recursive
   ```
3. Build for your platform:
   ```bash
   # macOS (Apple silicon)
   cmake -S native -B native/build/macos -DCMAKE_BUILD_TYPE=Release
   cmake --build native/build/macos
   export MLXSHARP_LIBRARY=$(pwd)/native/build/macos/libmlxsharp.dylib

   # Linux
   cmake -S native -B native/build/linux -DCMAKE_BUILD_TYPE=Release
   cmake --build native/build/linux
   export MLXSHARP_LIBRARY=$(pwd)/native/build/linux/libmlxsharp.so
   ```
4. Run your application or `dotnet pack` with explicit paths:
   ```bash
   dotnet pack src/MLXSharp/MLXSharp.csproj \
       -p:MLXSharpMacNativeBinary=$PWD/native/build/macos/libmlxsharp.dylib \
       -p:MLXSharpLinuxNativeBinary=$PWD/native/build/linux/libmlxsharp.so
   ```

The CMake project vendored from MLX builds MLX and the shim in one go. macOS builds enable Metal automatically; disable or tweak MLX options by passing flags such as `-DMLX_BUILD_METAL=OFF` or `-DMLX_BUILD_BLAS_FROM_SOURCE=ON`.

## CI overview
1. `dotnet-build` (Ubuntu): restores the solution and compiles managed projects.
2. `native-linux` / `native-macos`: compile `libmlxsharp.so` and `libmlxsharp.dylib` in parallel.
3. `package-test` (macOS): downloads both native artifacts, stages them into `src/MLXSharp/runtimes/{rid}/native`, rebuilds, runs the integration tests, and produces NuGet packages.

## Testing
Tests require a local MLX model bundle. Point `MLXSHARP_MODEL_PATH` to the directory before running:

```bash
export MLXSHARP_MODEL_PATH=$PWD/models/Qwen1.5-0.5B-Chat-4bit
huggingface-cli download mlx-community/Qwen1.5-0.5B-Chat-4bit --local-dir "$MLXSHARP_MODEL_PATH"
dotnet test
```

## Versioning & platform support
This initial release is focused on macOS developers who want MLX inside .NET applications. Linux binaries are produced to keep NuGet packages complete, and Windows support is not yet available.

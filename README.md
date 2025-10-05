# MLXSharp

MLXSharp offers .NET bindings and tooling around [Apple MLX](https://github.com/ml-explore/mlx) that plug directly into the [`Microsoft.Extensions.AI`](https://learn.microsoft.com/dotnet/ai/microsoft-extensions-ai) ecosystem.  
The design mirrors the packaging approach from projects such as [LLamaSharp](https://github.com/SciSharp/LLamaSharp): managed clients sit on top of a native wrapper that can be distributed through NuGet alongside prebuilt binaries.

## Features

- `IChatClient`, `IEmbeddingGenerator<string, Embedding<float)>`, and image generation helpers that adhere to the `Microsoft.Extensions.AI` abstractions.
- Builder-based backend configuration with a deterministic managed implementation for tests and a P/Invoke powered native backend.
- Native library resolver that probes application directories, `MLXSHARP_LIBRARY`, or packaged runtimes and loads `libmlxsharp` on demand.
- Dependency injection extensions (`AddMlx`) in **MLXSharp** package.
- Semantic Kernel integration (`AddMlxChatCompletion`) in separate **MLXSharp.SemanticKernel** package.
- Integration test suite that exercises chat, embedding, image, and Semantic Kernel flows against both managed and native backends.

## Repository Layout

```
├── extern/mlx                      # Git submodule with the official MLX sources
├── native/                         # Native wrapper scaffold (CMake project)
├── src/MLXSharp/                   # Managed library with Microsoft.Extensions.AI adapters + packaged runtimes
├── src/MLXSharp.SemanticKernel/    # Semantic Kernel integration (separate package)
└── src/MLXSharp.Tests/             # Integration tests covering DI and Semantic Kernel
```

## Prerequisites

The solution targets .NET 9 / C# 13 previews. Install the SDK via the official installer script:

```bash
curl -sSL https://dot.net/v1/dotnet-install.sh | bash /dev/stdin --version 9.0.305
export PATH="$PATH:$HOME/.dotnet"
dotnet --version
```

## Getting Started

Register MLX services through dependency injection:

```csharp
var services = new ServiceCollection();
services.AddMlx(builder =>
{
    builder.Configure(options =>
    {
        options.ChatModelId = "mlx-chat";
        options.EmbeddingModelId = "mlx-embedding";
    });

    // Managed fallback backend (default) that keeps tests deterministic.
    builder.UseManagedBackend(new MlxManagedBackend());

    // Or switch to the native backend once libmlxsharp is available.
    // builder.UseNativeBackend();
});
```

Semantic Kernel integration uses the same builder experience:

```csharp
var kernelBuilder = Kernel.CreateBuilder();
kernelBuilder.AddMlxChatCompletion(b => b.UseManagedBackend(new MlxManagedBackend()));
var kernel = kernelBuilder.Build();

var chat = kernel.Services.GetRequiredService<IChatCompletionService>();
var history = new ChatHistory();
history.AddUserMessage("Summarise MLX in one sentence");
var response = await chat.GetChatMessageContentsAsync(history, new PromptExecutionSettings(), kernel, CancellationToken.None);
```

## NuGet Packages

### MLXSharp [![NuGet](https://img.shields.io/nuget/v/MLXSharp.svg)](https://www.nuget.org/packages/MLXSharp)

Core package with `Microsoft.Extensions.AI` integration:

```bash
dotnet add package MLXSharp
```

This package contains:
- Managed DLL with `Microsoft.Extensions.AI` implementations
- Native assets in `runtimes/{rid}/native/`:
  - `runtimes/linux-x64/native/libmlxsharp.so.b64` - Base64-encoded stub that `MlxNativeLibrary` expands for CI/testing
  - `runtimes/osx-arm64/native/libmlxsharp.dylib` - built in CI on macOS

`MlxNativeLibrary` materialises `libmlxsharp.so` from the encoded payload on first use so Git history stays free of binary blobs while tests retain deterministic behaviour.

### MLXSharp.SemanticKernel [![NuGet](https://img.shields.io/nuget/v/MLXSharp.SemanticKernel.svg)](https://www.nuget.org/packages/MLXSharp.SemanticKernel)

Semantic Kernel integration:

```bash
dotnet add package MLXSharp.SemanticKernel
```

This package depends on MLXSharp and adds Semantic Kernel chat completion service.

### How It Works

GitHub Actions automatically:
1. Compiles native wrapper with MLX submodule
2. Copies `libmlxsharp.dylib` to `src/MLXSharp/runtimes/osx-arm64/native/`
3. Packs managed + native together in NuGet package

At runtime `MlxNativeLibrary` automatically finds the right library:
- Checks `MlxClientOptions.LibraryPath`
- Checks `MLXSHARP_LIBRARY` environment variable
- Searches in `runtimes/{rid}/native/` (NuGet unpacks automatically)
- Fallback to system library search paths

To build the native wrapper locally (mirroring the LLamaSharp workflow):

```bash
git submodule update --init --recursive
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
export MLXSHARP_LIBRARY=$(pwd)/native/build/libmlxsharp.dylib
```

The CMake project vendors the official [mlx](https://github.com/ml-explore/mlx) sources as a submodule and
builds them alongside `libmlxsharp`. On non-macOS hosts the configuration automatically disables the Metal backend;
pass `-DMLX_BUILD_METAL=ON` if you are compiling on Apple silicon and want GPU support. Additional MLX switches can be
forwarded on the command line, for example `-DMLX_BUILD_BLAS_FROM_SOURCE=ON` when targeting systems without an OpenBLAS
installation.

## Running Tests

```bash
dotnet test
```

The suite validates the managed backend, Semantic Kernel extensions, and the native loader by exercising the packaged stub library.

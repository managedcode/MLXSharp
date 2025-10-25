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
       -p:MLXSharpMacMetallibBinary=$PWD/native/build/macos/extern/mlx/mlx/backend/metal/kernels/mlx.metallib \
       -p:MLXSharpLinuxNativeBinary=$PWD/native/build/linux/libmlxsharp.so
   ```

The CMake project vendored from MLX builds MLX and the shim in one go. macOS builds enable Metal automatically; disable or tweak MLX options by passing flags such as `-DMLX_BUILD_METAL=OFF` or `-DMLX_BUILD_BLAS_FROM_SOURCE=ON`.

## CI overview
1. `dotnet-build` (Ubuntu): restores the solution and compiles managed projects.
2. `native-assets` (Ubuntu): downloads the signed native binaries published with the latest MLXSharp release and uploads them as workflow artifacts.
3. `package-test` (macOS): pulls down the staged native artifacts, copies them into `src/MLXSharp/runtimes/{rid}/native`, rebuilds, runs the integration tests, and produces NuGet packages.

## Testing
The managed integration tests still piggy-back on `mlx_lm` until the native runner is feature-complete. Bring your own HuggingFace bundle (any MLX-compatible repo) and point `MLXSHARP_MODEL_PATH` to it before running:

```bash
export MLXSHARP_HF_MODEL_ID=<your-mlx-model>
export MLXSHARP_MODEL_PATH=$PWD/models/<your-mlx-model>
huggingface-cli download "$MLXSHARP_HF_MODEL_ID" --local-dir "$MLXSHARP_MODEL_PATH"
python -m pip install mlx-lm
dotnet test
```

`MLXSHARP_HF_MODEL_ID` is picked up by the Python smoke test; omit it to fall back to `mlx-community/Qwen1.5-0.5B-Chat-4bit`.

When running locally you can place prebuilt binaries under `libs/native-osx-arm64` (and/or `libs/native-libs`) and a corresponding model bundle under `model/`. The test harness auto-discovers these folders and configures `MLXSHARP_LIBRARY`, `MLXSHARP_MODEL_PATH`, and `MLXSHARP_TOKENIZER_PATH` so you can iterate completely offline.

The integration suite invokes `python -m mlx_lm.generate` with deterministic settings (temperature `0`, seed `42`) and asserts that the generated response for prompts like “Скільки буде 2+2?” contains the correct answer. Test output includes the raw generation transcript so you can verify the model behaviour directly from the CI logs.

### Native pipeline (experimental)

Work is in progress to move inference fully into the native MLX backend. The current build exposes new configuration knobs via `MlxClientOptions`:

| Option | Description |
| --- | --- |
| `EnableNativeModelRunner` | Turns on the experimental native transformer pipeline. Still returns “not implemented” until the native side is completed. |
| `NativeModelDirectory` | Directory containing `config.json`, `*.safetensors`, etc. |
| `TokenizerPath` | Path to the HuggingFace `tokenizer.json` (loaded with `Microsoft.ML.Tokenizers`). |
| `MaxGeneratedTokens`, `Temperature`, `TopP`, `TopK` | Generation parameters that will flow into the native pipeline. |

When the C++ implementation catches up you’ll be able to set the environment variables below and exercise the path end-to-end:

```bash
export MLXSHARP_TOKENIZER_PATH=$PWD/models/<model-name>/tokenizer.json
export MLXSHARP_MODEL_PATH=$PWD/models/<model-name>
```

Until then, `EnableNativeModelRunner` should stay `false` to avoid runtime errors from the stub implementation.

### MSBuild properties

| Property | Purpose |
| --- | --- |
| `MLXSharpMacNativeBinary` | Path to `libmlxsharp.dylib` that gets packaged into the NuGet runtime folder. |
| `MLXSharpMacMetallibBinary` | Path to the matching `mlx.metallib` that ships next to the dylib. |
| `MLXSharpLinuxNativeBinary` | Path to the Linux shared object (`libmlxsharp.so`). |
| `MLXSharpSkipMacNativeValidation` / `MLXSharpSkipLinuxNativeValidation` | Opt-out flags for validation logic when you intentionally omit platform binaries. |

## Versioning & platform support
This initial release is focused on macOS developers who want MLX inside .NET applications. Linux binaries are produced to keep NuGet packages complete, and Windows support is not yet available.

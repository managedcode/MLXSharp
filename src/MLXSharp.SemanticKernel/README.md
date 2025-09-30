# MLXSharp.SemanticKernel

Semantic Kernel integration for MLXSharp - Apple MLX bindings for .NET.

## Installation

```bash
dotnet add package MLXSharp.SemanticKernel
```

## Usage

Add MLX chat completion to your Semantic Kernel:

```csharp
using Microsoft.SemanticKernel;
using MLXSharp.SemanticKernel;

var builder = Kernel.CreateBuilder();
builder.AddMlxChatCompletion(mlx =>
{
    mlx.Configure(options =>
    {
        options.ChatModelId = "path/to/mlx/model";
    });
    mlx.UseNativeBackend();
});

var kernel = builder.Build();

// Use chat completion service
var chat = kernel.Services.GetRequiredService<IChatCompletionService>();
var history = new ChatHistory();
history.AddUserMessage("Explain MLX in one sentence");

var result = await chat.GetChatMessageContentsAsync(history);
Console.WriteLine(result[0].Content);
```

## Dependencies

This package depends on:
- `MLXSharp` - Core MLX bindings
- `Microsoft.SemanticKernel` - Semantic Kernel framework

## Features

- Semantic Kernel chat completion integration
- Native MLX backend support
- Managed backend support for testing

## License

MIT
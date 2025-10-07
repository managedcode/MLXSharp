using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using MLXSharp;
using MLXSharp.Backends;
using Xunit;

namespace MLXSharp.Tests;

public sealed class ModelIntegrationTests
{
    [Fact]
    public async Task NativeBackendAnswersSimpleMathAsync()
    {
        TestEnvironment.EnsureInitialized();
        EnsureAssetsOrSkip();

        var options = CreateOptions();
        using var backend = MlxNativeBackend.Create(options);

        var request = new MlxTextRequest(
            new[] { new ChatMessage(ChatRole.User, "Скільки буде 2+2?") },
            new ChatOptions { Temperature = 0 });

        var result = await backend.GenerateTextAsync(request, CancellationToken.None);

        Assert.False(string.IsNullOrWhiteSpace(result.Text));
        Assert.Contains("4", result.Text);
    }

    private static MlxClientOptions CreateOptions()
    {
        var libraryPath = Environment.GetEnvironmentVariable("MLXSHARP_LIBRARY");
        var options = new MlxClientOptions
        {
            LibraryPath = string.IsNullOrWhiteSpace(libraryPath) ? null : libraryPath,
            EnableNativeModelRunner = false,
        };

        var modelId = Environment.GetEnvironmentVariable("MLXSHARP_HF_MODEL_ID");
        if (!string.IsNullOrWhiteSpace(modelId))
        {
            options.ChatModelId = modelId;
        }

        var modelDirectory = Environment.GetEnvironmentVariable("MLXSHARP_MODEL_PATH");
        if (!string.IsNullOrWhiteSpace(modelDirectory))
        {
            options.NativeModelDirectory = modelDirectory;
        }

        var tokenizerPath = Environment.GetEnvironmentVariable("MLXSHARP_TOKENIZER_PATH");
        if (!string.IsNullOrWhiteSpace(tokenizerPath))
        {
            options.TokenizerPath = tokenizerPath;
        }

        return options;
    }

    private static void EnsureAssetsOrSkip()
    {
        var modelPath = Environment.GetEnvironmentVariable("MLXSHARP_MODEL_PATH");
        if (string.IsNullOrWhiteSpace(modelPath) || !System.IO.Directory.Exists(modelPath))
        {
            Skip.If(true, "Native model bundle not found.");
        }

        var library = Environment.GetEnvironmentVariable("MLXSHARP_LIBRARY");
        if (string.IsNullOrWhiteSpace(library) || !System.IO.File.Exists(library))
        {
            Skip.If(true, "Native libmlxsharp library not configured.");
        }
    }
}

using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using MLXSharp;
using MLXSharp.Backends;
using Xunit;

namespace MLXSharp.Tests;

public sealed class ModelIntegrationTests
{
    [RequiresNativeModelFact]
    public async Task NativeBackendAnswersSimpleMathAsync()
    {
        TestEnvironment.EnsureInitialized();

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

}

internal sealed class RequiresNativeModelFactAttribute : FactAttribute
{
    public RequiresNativeModelFactAttribute()
    {
        TestEnvironment.EnsureInitialized();

        if (!NativeLibraryLocator.TryEnsure(out var skipReason))
        {
            Skip = skipReason ?? "Native MLX library is not available.";
            return;
        }

        var modelPath = Environment.GetEnvironmentVariable("MLXSHARP_MODEL_PATH");
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            Skip = "Native model bundle path is not configured. Set MLXSHARP_MODEL_PATH to a valid directory.";
            return;
        }

        if (!Directory.Exists(modelPath))
        {
            Skip = $"Native model bundle not found at '{modelPath}'.";
            return;
        }

        var library = Environment.GetEnvironmentVariable("MLXSHARP_LIBRARY");
        if (string.IsNullOrWhiteSpace(library) || !File.Exists(library))
        {
            Skip = "Native libmlxsharp library is not configured. Set MLXSHARP_LIBRARY to the staged native library that ships with the official MLXSharp release.";
            return;
        }

        var tokenizerPath = Environment.GetEnvironmentVariable("MLXSHARP_TOKENIZER_PATH");
        if (!string.IsNullOrWhiteSpace(tokenizerPath) && !File.Exists(tokenizerPath))
        {
            Skip = $"Native tokenizer file not found at '{tokenizerPath}'.";
        }
    }
}

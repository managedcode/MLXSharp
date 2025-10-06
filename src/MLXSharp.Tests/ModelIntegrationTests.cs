using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using MLXSharp;
using MLXSharp.Backends;
using Xunit;
using Xunit.Sdk;

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

        var result = await backend.GenerateTextAsync(request, CancellationToken.None).ConfigureAwait(false);

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

        return options;
    }

    private static void EnsureAssetsOrSkip()
    {
        var modelPath = Environment.GetEnvironmentVariable("MLXSHARP_MODEL_PATH");
        if (string.IsNullOrWhiteSpace(modelPath) || !System.IO.Directory.Exists(modelPath))
        {
            throw new SkipException("Native model bundle not found.");
        }

        var library = Environment.GetEnvironmentVariable("MLXSHARP_LIBRARY");
        if (string.IsNullOrWhiteSpace(library) || !System.IO.File.Exists(library))
        {
            throw new SkipException("Native libmlxsharp library not configured.");
        }
    }
}

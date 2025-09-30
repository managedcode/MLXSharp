using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using MLXSharp.Backends;
using MLXSharp.Clients;
using MLXSharp.DependencyInjection;
using MLXSharp.SemanticKernel;
using Xunit.Abstractions;

namespace MLXSharp.Tests;

public class MlxIntegrationTests(ITestOutputHelper output)
{
    [Fact]
    public async Task ChatClientGeneratesResponse()
    {
        var modelPath = GetRequiredModelPath();
        var nativeLibPath = GetRequiredNativeLibraryPath();

        var services = new ServiceCollection();
        services.AddMlx(builder =>
        {
            builder.Configure(options =>
            {
                options.ChatModelId = modelPath;
                options.LibraryPath = nativeLibPath;
            });
            builder.UseNativeBackend();
        });

        await using var provider = services.BuildServiceProvider();
        var chatClient = provider.GetRequiredService<IChatClient>();

        var response = await chatClient.GetResponseAsync(
            new[] { new ChatMessage(ChatRole.User, "What is 2+2?") },
            new ChatOptions { MaxOutputTokens = 50, Temperature = 0.7f },
            CancellationToken.None);

        Assert.NotEmpty(response.Messages);
        Assert.NotEmpty(response.Messages[0].Text);
        output.WriteLine($"Response: {response.Messages[0].Text}");
    }

    [Fact]
    public async Task MultipleRequestsWorkCorrectly()
    {
        var modelPath = GetRequiredModelPath();
        var nativeLibPath = GetRequiredNativeLibraryPath();

        var services = new ServiceCollection();
        services.AddMlx(builder =>
        {
            builder.Configure(options =>
            {
                options.ChatModelId = modelPath;
                options.LibraryPath = nativeLibPath;
            });
            builder.UseNativeBackend();
        });

        await using var provider = services.BuildServiceProvider();
        var chatClient = provider.GetRequiredService<IChatClient>();

        // First request
        var response1 = await chatClient.GetResponseAsync(
            new[] { new ChatMessage(ChatRole.User, "Say hello") },
            new ChatOptions { MaxOutputTokens = 30 },
            CancellationToken.None);

        Assert.NotEmpty(response1.Messages);
        output.WriteLine($"Response 1: {response1.Messages[0].Text}");

        // Second request
        var response2 = await chatClient.GetResponseAsync(
            new[] { new ChatMessage(ChatRole.User, "Count to 3") },
            new ChatOptions { MaxOutputTokens = 30 },
            CancellationToken.None);

        Assert.NotEmpty(response2.Messages);
        output.WriteLine($"Response 2: {response2.Messages[0].Text}");
    }

    [Fact]
    public async Task SemanticKernelIntegrationWorks()
    {
        var modelPath = GetRequiredModelPath();
        var nativeLibPath = GetRequiredNativeLibraryPath();

        var builder = Kernel.CreateBuilder();
        builder.AddMlxChatCompletion(b =>
        {
            b.Configure(options =>
            {
                options.ChatModelId = modelPath;
                options.LibraryPath = nativeLibPath;
            });
            b.UseNativeBackend();
        });

        var kernel = builder.Build();
        var chat = kernel.Services.GetRequiredService<IChatCompletionService>();
        var history = new ChatHistory();
        history.AddUserMessage("What is 1+1?");

        var result = await chat.GetChatMessageContentsAsync(
            history,
            new MlxPromptExecutionSettings { MaxTokens = 50 },
            kernel,
            CancellationToken.None);

        Assert.NotEmpty(result);
        Assert.NotNull(result[0].Content);
        Assert.False(string.IsNullOrEmpty(result[0].Content));
        output.WriteLine($"SK Response: {result[0].Content}");
    }

    [Fact]
    public async Task LongerConversationWithContext()
    {
        var modelPath = GetRequiredModelPath();
        var nativeLibPath = GetRequiredNativeLibraryPath();

        output.WriteLine($"Model: {modelPath}");
        output.WriteLine($"Library: {nativeLibPath}");

        var services = new ServiceCollection();
        services.AddMlx(builder =>
        {
            builder.Configure(options =>
            {
                options.ChatModelId = modelPath;
                options.LibraryPath = nativeLibPath;
            });
            builder.UseNativeBackend();
        });

        await using var provider = services.BuildServiceProvider();
        var chatClient = provider.GetRequiredService<IChatClient>();

        var messages = new List<ChatMessage>
        {
            new(ChatRole.User, "What is the capital of France?")
        };

        var stopwatch = System.Diagnostics.Stopwatch.StartNew();
        var response = await chatClient.GetResponseAsync(
            messages,
            new ChatOptions { MaxOutputTokens = 100, Temperature = 0.7f },
            CancellationToken.None);
        stopwatch.Stop();

        Assert.NotEmpty(response.Messages);
        Assert.NotEmpty(response.Messages[0].Text);

        output.WriteLine($"Response ({stopwatch.ElapsedMilliseconds}ms): {response.Messages[0].Text}");
    }

    private static string GetRequiredModelPath()
    {
        var modelPath = Environment.GetEnvironmentVariable("MLXSHARP_MODEL_PATH");
        Assert.False(string.IsNullOrEmpty(modelPath), "MLXSHARP_MODEL_PATH environment variable must be set");
        Assert.True(Directory.Exists(modelPath), $"Model directory does not exist: {modelPath}");
        return modelPath;
    }

    private static string GetRequiredNativeLibraryPath()
    {
        var baseDirectory = AppContext.BaseDirectory;
        var libraryName = OperatingSystem.IsWindows()
            ? "mlxsharp.dll"
            : OperatingSystem.IsMacOS()
                ? "libmlxsharp.dylib"
                : "libmlxsharp.so";
        var rid = OperatingSystem.IsMacOS()
            ? "osx-arm64"
            : OperatingSystem.IsWindows()
                ? "win-x64"
                : "linux-x64";
        var path = Path.Combine(baseDirectory, "runtimes", rid, "native", libraryName);
        Assert.True(File.Exists(path), $"Native library not found: {path}");
        return path;
    }
}

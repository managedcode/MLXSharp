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

namespace MLXSharp.Tests;

public class MlxIntegrationTests
{
    [Fact]
    public async Task ServiceCollectionProvidesEndToEndClients()
    {
        var services = new ServiceCollection();
        services.AddMlx(builder =>
        {
            builder.Configure(options => options.ChatModelId = "test-chat");
        });

        await using var provider = services.BuildServiceProvider();
        var chatClient = provider.GetRequiredService<IChatClient>();
        var embeddingGenerator = provider.GetRequiredService<IEmbeddingGenerator<string, Embedding<float>>>();
        var imageClient = provider.GetRequiredService<IMlxImageClient>();

        var chatHistory = new List<ChatMessage>
        {
            new(ChatRole.User, "привіт"),
        };

        var response = await chatClient.GetResponseAsync(chatHistory, new ChatOptions { Temperature = 0.2f }, CancellationToken.None);
        Assert.NotEmpty(response.Messages);
        Assert.Contains("user:привіт", response.Messages[0].Text);
        Assert.Equal("test-chat", response.ModelId);

        var embeddings = await embeddingGenerator.GenerateAsync(new[] { "mlx" }, null, CancellationToken.None);
        Assert.Single(embeddings);
        Assert.Equal(8, embeddings[0].Vector.Length);

        var image = await imageClient.GenerateImageAsync("apple mlx");
        Assert.Equal("image/png", image.MediaType);
        Assert.True(image.Data.Length > 0);
    }

    [Fact]
    public async Task NativeBackendLoadsStubLibrary()
    {
        var services = new ServiceCollection();
        services.AddMlx(builder =>
        {
            builder.Configure(options =>
            {
                options.ChatModelId = "native-stub";
                options.LibraryPath = ResolveNativeLibraryPath();
            });
            builder.UseNativeBackend();
        });

        await using var provider = services.BuildServiceProvider();
        var chatClient = provider.GetRequiredService<IChatClient>();
        var response = await chatClient.GetResponseAsync(new[] { new ChatMessage(ChatRole.User, "ping") }, null, CancellationToken.None);

        Assert.NotEmpty(response.Messages);
        Assert.Contains("mlxstub", response.Messages[0].Text, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public async Task SemanticKernelExtensionAddsChatService()
    {
        var builder = Kernel.CreateBuilder();
        builder.AddMlxChatCompletion(b =>
        {
            b.Configure(options => options.ChatModelId = "sk-model");
            b.UseManagedBackend(new MlxManagedBackend());
        });

        var kernel = builder.Build();
        var chat = kernel.Services.GetRequiredService<IChatCompletionService>();
        var history = new ChatHistory();
        history.AddUserMessage("Explain MLX in one sentence");

        var result = await chat.GetChatMessageContentsAsync(history, new PromptExecutionSettings(), kernel, CancellationToken.None);
        Assert.NotEmpty(result);
        Assert.Contains("mlx", result[0].Content, StringComparison.OrdinalIgnoreCase);
    }

    [Fact(Skip = "Requires real MLX model and native library")]
    public async Task RealModelGeneratesText()
    {
        var modelPath = Environment.GetEnvironmentVariable("MLXSHARP_MODEL_PATH");
        if (string.IsNullOrEmpty(modelPath) || !Directory.Exists(modelPath))
        {
            // Skip if model not available
            return;
        }

        var services = new ServiceCollection();
        services.AddMlx(builder =>
        {
            builder.Configure(options =>
            {
                options.ChatModelId = modelPath;
                options.LibraryPath = ResolveNativeLibraryPath();
            });
            builder.UseNativeBackend();
        });

        await using var provider = services.BuildServiceProvider();
        var chatClient = provider.GetRequiredService<IChatClient>();

        var response = await chatClient.GetResponseAsync(
            new[] { new ChatMessage(ChatRole.User, "Say hello in Ukrainian") },
            new ChatOptions { MaxOutputTokens = 50 },
            CancellationToken.None);

        Assert.NotEmpty(response.Messages);
        Assert.NotEmpty(response.Messages[0].Text);
        // Should contain Ukrainian greeting
        Assert.True(response.Messages[0].Text.Contains("Привіт", StringComparison.OrdinalIgnoreCase) ||
                   response.Messages[0].Text.Contains("Вітаю", StringComparison.OrdinalIgnoreCase) ||
                   response.Messages[0].Text.Contains("Здрастуйте", StringComparison.OrdinalIgnoreCase),
                   $"Expected Ukrainian greeting but got: {response.Messages[0].Text}");
    }

    private static string? ResolveNativeLibraryPath()
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
        var candidate = Path.Combine(baseDirectory, "runtimes", rid, "native", libraryName);
        return File.Exists(candidate) ? candidate : null;
    }
}

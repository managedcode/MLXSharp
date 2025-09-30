using System;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;
using MLXSharp.Backends;
using MLXSharp.DependencyInjection;

namespace MLXSharp.SemanticKernel;

public static class MlxKernelBuilderExtensions
{
    public static IKernelBuilder AddMlxChatCompletion(this IKernelBuilder builder, Action<MlxClientBuilder>? configure = null)
    {
        ArgumentNullException.ThrowIfNull(builder);
        builder.Services.AddMlx(configure);
        builder.Services.AddSingleton<IChatCompletionService>(sp =>
        {
            var chatClient = sp.GetRequiredService<IChatClient>();
            return chatClient.AsChatCompletionService(sp);
        });
        return builder;
    }
}

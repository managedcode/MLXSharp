using System;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using MLXSharp.Backends;
using MLXSharp.Clients;

namespace MLXSharp.DependencyInjection;

public static class MlxServiceCollectionExtensions
{
    public static IServiceCollection AddMlx(this IServiceCollection services, Action<MlxClientBuilder>? configure = null)
    {
        ArgumentNullException.ThrowIfNull(services);
        var builder = new MlxClientBuilder();
        configure?.Invoke(builder);

        services.TryAddSingleton(builder.Options);
        services.TryAddSingleton<IMlxBackend>(sp => builder.BuildBackendFactory()(sp));
        services.TryAddSingleton<IChatClient, MlxChatClient>();
        services.TryAddSingleton<IEmbeddingGenerator<string, Embedding<float>>, MlxEmbeddingGenerator>();
        services.TryAddSingleton<IMlxImageClient, MlxImageClient>();

        return services;
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using MLXSharp.Backends;

namespace MLXSharp.Clients;

public sealed class MlxEmbeddingGenerator : IEmbeddingGenerator<string, Embedding<float>>
{
    private readonly IMlxBackend _backend;
    private readonly EmbeddingGeneratorMetadata _metadata;

    public MlxEmbeddingGenerator(IMlxBackend backend, MlxClientOptions options)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        ArgumentNullException.ThrowIfNull(options);
        _metadata = new EmbeddingGeneratorMetadata(options.ProviderName, options.ProviderUri, options.EmbeddingModelId, default);
    }

    public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(IEnumerable<string> values, EmbeddingGenerationOptions? options, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(values);
        var input = values.ToList();
        var result = await _backend.GenerateEmbeddingAsync(new MlxEmbeddingRequest(input, options), cancellationToken).ConfigureAwait(false);
        var generated = new GeneratedEmbeddings<Embedding<float>>(result.Embeddings);
        if (result.Usage is { } usage)
        {
            generated.Usage = usage;
        }
        if (!string.IsNullOrWhiteSpace(result.ModelId))
        {
            generated.AdditionalProperties ??= new AdditionalPropertiesDictionary();
            generated.AdditionalProperties["modelId"] = result.ModelId!;
        }

        return generated;
    }

    public void Dispose()
    {
    }

    public object? GetService(Type serviceType, object? serviceKey)
    {
        if (serviceType == typeof(EmbeddingGeneratorMetadata))
        {
            return _metadata;
        }

        if (serviceType.IsInstanceOfType(this))
        {
            return this;
        }

        return null;
    }
}

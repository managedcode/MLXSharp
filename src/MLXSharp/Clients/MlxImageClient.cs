using System;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using MLXSharp.Backends;

namespace MLXSharp.Clients;

public sealed class MlxImageClient : IMlxImageClient
{
    private readonly IMlxBackend _backend;
    private readonly MlxClientOptions _options;

    public MlxImageClient(IMlxBackend backend, MlxClientOptions options)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public async Task<DataContent> GenerateImageAsync(string prompt, MlxImageOptions? options = null, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(prompt);
        options ??= new MlxImageOptions();
        var request = new MlxImageRequest(prompt, options.Width, options.Height, options.MediaType);
        var result = await _backend.GenerateImageAsync(request, cancellationToken).ConfigureAwait(false);
        var content = new DataContent(result.Data, result.MediaType);
        content.AdditionalProperties ??= new AdditionalPropertiesDictionary();
        var modelId = result.ModelId ?? _options.ImageModelId;
        if (!string.IsNullOrEmpty(modelId))
        {
            content.AdditionalProperties["modelId"] = modelId;
        }

        if (result.Usage is { } usage)
        {
            content.AdditionalProperties["usage"] = usage;
        }

        return content;
    }
}

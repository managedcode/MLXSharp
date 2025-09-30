using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using MLXSharp.Backends;

namespace MLXSharp.Clients;

public sealed class MlxChatClient : IChatClient
{
    private readonly IMlxBackend _backend;
    private readonly ChatClientMetadata _metadata;

    public MlxChatClient(IMlxBackend backend, MlxClientOptions options)
    {
        _backend = backend ?? throw new ArgumentNullException(nameof(backend));
        ArgumentNullException.ThrowIfNull(options);
        _metadata = new ChatClientMetadata(options.ProviderName, options.ProviderUri, options.ChatModelId);
    }

    public async Task<ChatResponse> GetResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var history = messages.ToList();
        var result = await _backend.GenerateTextAsync(new MlxTextRequest(history, options), cancellationToken).ConfigureAwait(false);
        return CreateResponse(result);
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<ChatMessage> messages, ChatOptions? options, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(messages);
        var history = messages.ToList();
        await foreach (var update in _backend.GenerateTextStreamingAsync(new MlxTextRequest(history, options), cancellationToken).ConfigureAwait(false))
        {
            yield return new ChatResponseUpdate(ChatRole.Assistant, new List<AIContent> { new TextContent(update.Delta) });
            if (update.IsCompleted)
            {
                yield break;
            }
        }
    }

    public object? GetService(Type serviceType, object? serviceKey)
    {
        if (serviceType == typeof(ChatClientMetadata))
        {
            return _metadata;
        }

        if (serviceType.IsInstanceOfType(this))
        {
            return this;
        }

        return null;
    }

    public void Dispose()
    {
    }

    private static ChatResponse CreateResponse(MlxTextResult result)
    {
        var response = new ChatResponse();
        response.Messages.Add(new ChatMessage(ChatRole.Assistant, new List<AIContent> { new TextContent(result.Text) }));
        response.ModelId = result.ModelId;
        if (result.Usage is { } usage)
        {
            response.Usage = usage;
        }

        return response;
    }
}

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;

namespace MLXSharp.Backends;

public sealed class MlxManagedBackend : IMlxBackend
{
    private readonly MlxClientOptions _options;

    public MlxManagedBackend()
        : this(new MlxClientOptions())
    {
    }

    public MlxManagedBackend(MlxClientOptions options)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public Task<MlxTextResult> GenerateTextAsync(MlxTextRequest request, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        var builder = new StringBuilder();
        foreach (var message in request.Messages)
        {
            if (message.Role == ChatRole.User)
            {
                builder.AppendLine($"user:{message.Text}");
            }
            else if (message.Role == ChatRole.System)
            {
                builder.AppendLine($"system:{message.Text}");
            }
        }

        builder.AppendLine($"mlx:{request.Options?.Temperature ?? 0:0.00}");
        var text = builder.ToString().TrimEnd();

        var usage = new UsageDetails
        {
            InputTokenCount = request.Messages.Count,
            OutputTokenCount = text.Length,
        };

        return Task.FromResult(new MlxTextResult(text, usage, _options.ChatModelId));
    }

    public async IAsyncEnumerable<MlxTextStreamingUpdate> GenerateTextStreamingAsync(MlxTextRequest request, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var result = await GenerateTextAsync(request, cancellationToken).ConfigureAwait(false);
        var chunkSize = Math.Max(1, result.Text.Length / 4);
        for (var i = 0; i < result.Text.Length; i += chunkSize)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var length = Math.Min(chunkSize, result.Text.Length - i);
            yield return new MlxTextStreamingUpdate(result.Text.Substring(i, length), false);
        }

        yield return new MlxTextStreamingUpdate(string.Empty, true);
    }

    public Task<MlxEmbeddingResult> GenerateEmbeddingAsync(MlxEmbeddingRequest request, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        var embeddings = new List<Embedding<float>>(request.Values.Count);
        foreach (var value in request.Values)
        {
            cancellationToken.ThrowIfCancellationRequested();
            embeddings.Add(CreateEmbedding(value));
        }

        var usage = new UsageDetails
        {
            InputTokenCount = request.Values.Count,
            OutputTokenCount = 0,
        };

        return Task.FromResult(new MlxEmbeddingResult(embeddings, usage, _options.EmbeddingModelId));
    }

    public Task<MlxImageResult> GenerateImageAsync(MlxImageRequest request, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(request);
        using var sha = SHA256.Create();
        var bytes = sha.ComputeHash(Encoding.UTF8.GetBytes($"{request.Prompt}:{request.Width}x{request.Height}:{request.MediaType}"));
        var usage = new UsageDetails
        {
            InputTokenCount = request.Prompt.Length,
            OutputTokenCount = bytes.Length,
        };
        return Task.FromResult(new MlxImageResult(new ReadOnlyMemory<byte>(bytes), request.MediaType, _options.ImageModelId, usage));
    }

    private static Embedding<float> CreateEmbedding(string value)
    {
        var bytes = Encoding.UTF8.GetBytes(value ?? string.Empty);
        const int dimensions = 8;
        var vector = ArrayPool<float>.Shared.Rent(dimensions);
        try
        {
            Array.Clear(vector, 0, dimensions);
            for (var i = 0; i < bytes.Length; i++)
            {
                vector[i % dimensions] += bytes[i] / 255f;
            }

            var slice = new float[dimensions];
            Array.Copy(vector, slice, dimensions);
            return new Embedding<float>(slice)
            {
                ModelId = "mlx-managed",
            };
        }
        finally
        {
            ArrayPool<float>.Shared.Return(vector);
        }
    }
}

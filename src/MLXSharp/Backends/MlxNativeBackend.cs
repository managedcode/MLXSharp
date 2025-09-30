using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;
using MLXSharp.Native;

namespace MLXSharp.Backends;

public sealed class MlxNativeBackend : IMlxBackend, IDisposable
{
    private readonly MlxClientOptions _options;
    private readonly SafeMlxSessionHandle _session;
    private bool _disposed;

    private MlxNativeBackend(SafeMlxSessionHandle session, MlxClientOptions options)
    {
        _session = session;
        _options = options;
    }

    public static MlxNativeBackend Create(MlxClientOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        MlxNativeLibrary.EnsureLoaded(options.LibraryPath);

        var status = MlxNativeMethods.CreateSession(options.ChatModelId, options.EmbeddingModelId, options.ImageModelId, out var session);
        if (status != 0 || session.IsInvalid)
        {
            session.Dispose();
            throw new InvalidOperationException($"Failed to create MLX session. Native status: {status}.");
        }

        return new MlxNativeBackend(session, options);
    }

    public Task<MlxTextResult> GenerateTextAsync(MlxTextRequest request, CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        var prompt = BuildPrompt(request.Messages);
        var status = MlxNativeMethods.GenerateText(_session, prompt, out var responseHandle, out var usage);
        if (status != 0)
        {
            responseHandle.Dispose();
            throw new InvalidOperationException($"mlxsharp_generate_text failed with status {status}.");
        }

        using (responseHandle)
        {
            var text = responseHandle.GetString();
            var usageDetails = CreateUsage(usage);
            return Task.FromResult(new MlxTextResult(text, usageDetails, _options.ChatModelId));
        }
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
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        var results = new List<Embedding<float>>(request.Values.Count);
        UsageDetails? usageDetails = null;
        foreach (var value in request.Values)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var status = MlxNativeMethods.GenerateEmbedding(_session, value ?? string.Empty, out var pointer, out var dimension, out var usage);
            if (status != 0)
            {
                if (pointer != nint.Zero)
                {
                    MlxNativeMethods.FreeEmbedding(pointer);
                }

                throw new InvalidOperationException($"mlxsharp_generate_embedding failed with status {status}.");
            }

            float[] vector = Array.Empty<float>();
            if (pointer != nint.Zero && dimension > 0)
            {
                vector = new float[dimension];
                Marshal.Copy(pointer, vector, 0, dimension);
                MlxNativeMethods.FreeEmbedding(pointer);
            }

            results.Add(new Embedding<float>(vector)
            {
                ModelId = _options.EmbeddingModelId,
            });

            usageDetails = CombineUsage(usageDetails, usage);
        }

        return Task.FromResult(new MlxEmbeddingResult(results, usageDetails, _options.EmbeddingModelId));
    }

    public Task<MlxImageResult> GenerateImageAsync(MlxImageRequest request, CancellationToken cancellationToken)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(request);

        var status = MlxNativeMethods.GenerateImage(_session, request.Prompt, request.Width, request.Height, out var pointer, out var length, out var usage);
        if (status != 0)
        {
            if (pointer != nint.Zero)
            {
                MlxNativeMethods.FreeBuffer(pointer);
            }

            throw new InvalidOperationException($"mlxsharp_generate_image failed with status {status}.");
        }

        ReadOnlyMemory<byte> data = ReadOnlyMemory<byte>.Empty;
        if (pointer != nint.Zero && length > 0)
        {
            var buffer = new byte[length];
            Marshal.Copy(pointer, buffer, 0, length);
            data = new ReadOnlyMemory<byte>(buffer);
            MlxNativeMethods.FreeBuffer(pointer);
        }

        var usageDetails = CreateUsage(usage);
        return Task.FromResult(new MlxImageResult(data, request.MediaType, _options.ImageModelId, usageDetails));
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        _session.Dispose();
        _disposed = true;
    }

    private static string BuildPrompt(IReadOnlyList<ChatMessage> messages)
    {
        var builder = new StringBuilder();
        foreach (var message in messages)
        {
            var role = message.Role.ToString();
            if (!string.IsNullOrWhiteSpace(role))
            {
                builder.Append(role.ToLowerInvariant());
                builder.Append(":");
            }

            builder.AppendLine(message.Text ?? string.Empty);
        }

        return builder.ToString();
    }

    private static UsageDetails? CreateUsage(in MlxUsage usage)
    {
        if (usage.InputTokens == 0 && usage.OutputTokens == 0)
        {
            return null;
        }

        return new UsageDetails
        {
            InputTokenCount = usage.InputTokens,
            OutputTokenCount = usage.OutputTokens,
        };
    }

    private static UsageDetails? CombineUsage(UsageDetails? existing, in MlxUsage next)
    {
        if (existing is null)
        {
            return CreateUsage(next);
        }

        existing.InputTokenCount += next.InputTokens;
        existing.OutputTokenCount += next.OutputTokens;
        return existing;
    }

    private void ThrowIfDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(MlxNativeBackend));
        }
    }
}

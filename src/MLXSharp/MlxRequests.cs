using System;
using System.Collections.Generic;
using Microsoft.Extensions.AI;

namespace MLXSharp;

public sealed record MlxTextRequest(IReadOnlyList<ChatMessage> Messages, ChatOptions? Options);

public sealed record MlxTextResult(string Text, UsageDetails? Usage = null, string? ModelId = null);

public sealed record MlxTextStreamingUpdate(string Delta, bool IsCompleted);

public sealed record MlxEmbeddingRequest(IReadOnlyList<string> Values, EmbeddingGenerationOptions? Options);

public sealed record MlxEmbeddingResult(IReadOnlyList<Embedding<float>> Embeddings, UsageDetails? Usage = null, string? ModelId = null);

public sealed record MlxImageRequest(string Prompt, int Width, int Height, string MediaType);

public sealed record MlxImageResult(ReadOnlyMemory<byte> Data, string MediaType, string? ModelId = null, UsageDetails? Usage = null);

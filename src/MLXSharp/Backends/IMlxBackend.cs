using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace MLXSharp.Backends;

public interface IMlxBackend
{
    Task<MlxTextResult> GenerateTextAsync(MlxTextRequest request, CancellationToken cancellationToken);

    IAsyncEnumerable<MlxTextStreamingUpdate> GenerateTextStreamingAsync(MlxTextRequest request, CancellationToken cancellationToken);

    Task<MlxEmbeddingResult> GenerateEmbeddingAsync(MlxEmbeddingRequest request, CancellationToken cancellationToken);

    Task<MlxImageResult> GenerateImageAsync(MlxImageRequest request, CancellationToken cancellationToken);
}

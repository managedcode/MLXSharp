using System;

namespace MLXSharp;

/// <summary>
/// Common options that describe the MLX backend and the default models to use.
/// </summary>
public sealed class MlxClientOptions
{
    public string ProviderName { get; set; } = "Apple MLX";

    public Uri ProviderUri { get; set; } = new("https://github.com/ml-explore/mlx");

    public string ChatModelId { get; set; } = "mlx-default-chat";

    public string EmbeddingModelId { get; set; } = "mlx-default-embedding";

    public string ImageModelId { get; set; } = "mlx-default-image";

    public string? LibraryPath { get; set; }
        = OperatingSystem.IsMacOS() ? "/usr/local/lib/libmlxsharp.dylib" : null;
}

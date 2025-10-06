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

    /// <summary>
    /// Optional directory containing native model assets (e.g. safetensors, config.json).
    /// </summary>
    public string? NativeModelDirectory { get; set; }

    /// <summary>
    /// Optional path to the tokenizer.json file matching <see cref="NativeModelDirectory"/>.
    /// </summary>
    public string? TokenizerPath { get; set; }

    /// <summary>
    /// Enables the experimental native inference path. When false the backend falls back to the
    /// legacy stub generation logic until the native path is fully implemented.
    /// </summary>
    public bool EnableNativeModelRunner { get; set; }

    public int MaxGeneratedTokens { get; set; } = 512;

    public float Temperature { get; set; } = 0.0f;

    public float TopP { get; set; } = 1.0f;

    public int TopK
    {
        get => _topK;
        set => _topK = value < 0 ? 0 : value;
    }

    private int _topK;
}

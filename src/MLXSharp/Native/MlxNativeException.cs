using System;

namespace MLXSharp.Native;

internal sealed class MlxNativeException : Exception
{
    public MlxNativeException(MlxStatus status, string message)
        : base(string.IsNullOrWhiteSpace(message) ? $"Native MLX call failed with status {(int)status}." : message)
    {
        Status = status;
    }

    public MlxStatus Status { get; }
}


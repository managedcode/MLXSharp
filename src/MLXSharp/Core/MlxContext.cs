using System;
using MLXSharp.Native;

namespace MLXSharp.Core;

/// <summary>
/// Represents a handle to an MLX execution context (device + stream).
/// </summary>
public sealed class MlxContext : IDisposable
{
    private bool _disposed;

    private MlxContext(SafeMlxContextHandle handle, MlxDeviceKind kind, int deviceIndex)
    {
        Handle = handle;
        DeviceKind = kind;
        DeviceIndex = deviceIndex;
    }

    internal SafeMlxContextHandle Handle { get; }

    public MlxDeviceKind DeviceKind { get; }

    public int DeviceIndex { get; }

    public static MlxContext CreateCpu(int deviceIndex = 0)
        => Create(MlxDeviceKind.Cpu, deviceIndex);

    public static MlxContext CreateGpu(int deviceIndex = 0)
        => Create(MlxDeviceKind.Gpu, deviceIndex);

    public static MlxContext Create(MlxDeviceKind kind, int deviceIndex = 0)
    {
        MlxNativeLibrary.EnsureLoaded(explicitPath: null);
        var status = MlxNativeMethods.ContextCreate(kind, deviceIndex, out var handle);
        status.ThrowIfFailed();
        return new MlxContext(handle, kind, deviceIndex);
    }

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        Handle.Dispose();
        _disposed = true;
    }
}

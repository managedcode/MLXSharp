using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.InteropServices;
using MLXSharp.Core;
using Xunit;

namespace MLXSharp.Tests;

public sealed class ArraySmokeTests
{
    [RequiresNativeLibraryFact]
    public void AddTwoFloatArrays()
    {
        using var context = MlxContext.CreateCpu();

        ReadOnlySpan<float> leftData = stackalloc float[] { 1f, 2f, 3f, 4f };
        ReadOnlySpan<float> rightData = stackalloc float[] { 5f, 6f, 7f, 8f };
        ReadOnlySpan<long> shape = stackalloc long[] { 2, 2 };

        using var left = MlxArray.From(context, leftData, shape);
        using var right = MlxArray.From(context, rightData, shape);
        using var result = MlxArray.Add(left, right);

        Assert.Equal(new[] { 6f, 8f, 10f, 12f }, result.ToArrayFloat32());
        Assert.Equal(shape.ToArray(), result.Shape);
        Assert.Equal(MlxDType.Float32, result.DType);
    }

    [RequiresNativeLibraryFact]
    public void ZerosAllocatesRequestedShape()
    {
        using var context = MlxContext.CreateCpu();
        ReadOnlySpan<long> shape = stackalloc long[] { 3, 1 };

        using var zeros = MlxArray.Zeros(context, shape, MlxDType.Float32);

        Assert.Equal(MlxDType.Float32, zeros.DType);
        Assert.Equal(shape.ToArray(), zeros.Shape);
        Assert.All(zeros.ToArrayFloat32(), value => Assert.Equal(0f, value));
    }
}

internal sealed class RequiresNativeLibraryFactAttribute : FactAttribute
{
    public RequiresNativeLibraryFactAttribute()
    {
        TestEnvironment.EnsureInitialized();
        if (!NativeLibraryLocator.TryEnsure(out var skipReason))
        {
            Skip = skipReason ?? "Native MLX library is not available.";
        }
    }
}

internal static class NativeLibraryLocator
{
    private static readonly object s_sync = new();
    private static bool s_initialized;
    private static bool s_available;

    public static bool TryEnsure(out string? skipReason)
    {
        lock (s_sync)
        {
            if (s_initialized)
            {
                skipReason = s_available ? null : "Native MLX library is not available.";
                return s_available;
            }

            if (!TryFindNativeLibrary(out var path))
            {
                s_initialized = true;
                s_available = false;
                skipReason = "Native MLX library is not available. Build the native project first.";
                return false;
            }

            if (!HasRequiredExports(path, out skipReason))
            {
                s_initialized = true;
                s_available = false;
                return false;
            }

            Environment.SetEnvironmentVariable("MLXSHARP_LIBRARY", path);
            s_initialized = true;
            s_available = true;
            skipReason = null;
            return true;
        }
    }

    private static bool HasRequiredExports(string path, out string? reason)
    {
        if (!NativeLibrary.TryLoad(path, out var handle))
        {
            reason = $"Unable to load native library from '{path}'.";
            return false;
        }

        try
        {
            foreach (var export in new[] { "mlxsharp_context_create", "mlxsharp_array_from_buffer", "mlxsharp_generate_text" })
            {
                if (!NativeLibrary.TryGetExport(handle, export, out _))
                {
                    reason = $"Native library at '{path}' is missing required export '{export}'. Rebuild MLXSharp native binaries.";
                    return false;
                }
            }

            reason = null;
            return true;
        }
        finally
        {
            NativeLibrary.Free(handle);
        }
    }

    private static bool TryFindNativeLibrary(out string path)
    {
        var baseDir = AppContext.BaseDirectory;
        var libraryName = OperatingSystem.IsWindows()
            ? "mlxsharp.dll"
            : OperatingSystem.IsMacOS()
                ? "libmlxsharp.dylib"
                : "libmlxsharp.so";

        foreach (var candidate in EnumerateCandidates(baseDir, libraryName))
        {
            if (File.Exists(candidate))
            {
                path = candidate;
                return true;
            }
        }

        path = string.Empty;
        return false;
    }

    private static IEnumerable<string> EnumerateCandidates(string baseDir, string libraryName)
    {
        var arch = System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture switch
        {
            System.Runtime.InteropServices.Architecture.Arm64 => "arm64",
            System.Runtime.InteropServices.Architecture.X64 => "x64",
            _ => string.Empty,
        };

        if (!string.IsNullOrEmpty(arch))
        {
            var rid = OperatingSystem.IsMacOS()
                ? $"osx-{arch}"
                : OperatingSystem.IsLinux()
                    ? $"linux-{arch}"
                    : OperatingSystem.IsWindows()
                        ? $"win-{arch}"
                        : string.Empty;

            if (!string.IsNullOrEmpty(rid))
            {
                yield return Path.Combine(baseDir, "runtimes", rid, "native", libraryName);
            }
        }

        yield return Path.Combine(baseDir, libraryName);
    }
}

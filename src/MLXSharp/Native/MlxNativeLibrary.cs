using System;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;

namespace MLXSharp.Native;

internal static class MlxNativeLibrary
{
    private static readonly object s_sync = new();
    private static bool s_loaded;
    private static nint s_handle;
    private static string? s_lastExplicitPath;

    static MlxNativeLibrary()
    {
        NativeLibrary.SetDllImportResolver(typeof(MlxNativeLibrary).Assembly, ResolveLibrary);
    }

    private static string LibraryFileName => OperatingSystem.IsWindows()
        ? "mlxsharp.dll"
        : OperatingSystem.IsMacOS()
            ? "libmlxsharp.dylib"
            : "libmlxsharp.so";

    public static void EnsureLoaded(string? explicitPath)
    {
        lock (s_sync)
        {
            s_lastExplicitPath = explicitPath;
            if (s_loaded)
            {
                return;
            }

            if (!TryLoad(explicitPath, out var handle))
            {
                throw new DllNotFoundException($"Unable to locate {LibraryFileName}. Set MLXSHARP_LIBRARY or copy the native binary next to the application.");
            }

            s_handle = handle;
            s_loaded = true;
        }
    }

    public static bool TryLoad(string? explicitPath, out nint handle)
    {
        foreach (var candidate in EnumerateProbePaths(explicitPath))
        {
            if (string.IsNullOrWhiteSpace(candidate))
            {
                continue;
            }

            var materialized = EnsureStubMaterialized(candidate);

            if (NativeLibrary.TryLoad(materialized, out handle))
            {
                return true;
            }
        }

        handle = nint.Zero;
        return false;
    }

    private static IEnumerable<string?> EnumerateProbePaths(string? explicitPath)
    {
        yield return explicitPath;
        yield return Environment.GetEnvironmentVariable("MLXSHARP_LIBRARY");

        var baseDirectory = AppContext.BaseDirectory;
        var runtimeId = GetRuntimeIdentifier();
        if (!string.IsNullOrWhiteSpace(baseDirectory) && runtimeId is not null)
        {
            yield return Path.Combine(baseDirectory, "runtimes", runtimeId, "native", LibraryFileName);
        }

        if (!string.IsNullOrWhiteSpace(baseDirectory))
        {
            yield return Path.Combine(baseDirectory, LibraryFileName);
        }

        yield return LibraryFileName;
    }

    private static string? GetRuntimeIdentifier()
    {
        var arch = RuntimeInformation.ProcessArchitecture switch
        {
            Architecture.Arm64 => "arm64",
            Architecture.X64 => "x64",
            _ => null,
        };

        if (arch is null)
        {
            return null;
        }

        if (OperatingSystem.IsMacOS())
        {
            return $"osx-{arch}";
        }

        if (OperatingSystem.IsLinux())
        {
            return $"linux-{arch}";
        }

        if (OperatingSystem.IsWindows())
        {
            return $"win-{arch}";
        }

        return null;
    }

    private static nint ResolveLibrary(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (!libraryName.Equals("libmlxsharp", StringComparison.Ordinal) && !libraryName.Equals("mlxsharp", StringComparison.Ordinal))
        {
            return nint.Zero;
        }

        lock (s_sync)
        {
            if (s_loaded && s_handle != nint.Zero)
            {
                return s_handle;
            }

            if (!TryLoad(s_lastExplicitPath, out var handle))
            {
                return nint.Zero;
            }

            s_handle = handle;
            s_loaded = true;
            return s_handle;
        }
    }

    private static string EnsureStubMaterialized(string candidate)
    {
        if (File.Exists(candidate))
        {
            return candidate;
        }

        var encodedPath = candidate + ".b64";
        if (!File.Exists(encodedPath))
        {
            return candidate;
        }

        lock (s_sync)
        {
            if (File.Exists(candidate))
            {
                return candidate;
            }

            var base64 = File.ReadAllText(encodedPath);
            var bytes = Convert.FromBase64String(base64);
            var directory = Path.GetDirectoryName(candidate);
            if (!string.IsNullOrEmpty(directory))
            {
                Directory.CreateDirectory(directory);
            }

            File.WriteAllBytes(candidate, bytes);
        }

        return candidate;
    }
}

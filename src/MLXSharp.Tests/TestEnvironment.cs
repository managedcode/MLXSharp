using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;

namespace MLXSharp.Tests;

internal static class TestEnvironment
{
    private static int s_initialized;

    public static void EnsureInitialized()
    {
        if (Interlocked.Exchange(ref s_initialized, 1) != 0)
        {
            return;
        }

        var baseDirectory = AppContext.BaseDirectory;
        var repoRoot = Path.GetFullPath(Path.Combine(baseDirectory, "..", "..", "..", ".."));

        ConfigureNativeLibrary(repoRoot);
        ConfigureModelPaths(repoRoot);
    }

    private static void ConfigureNativeLibrary(string repoRoot)
    {
        if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MLXSHARP_LIBRARY")))
        {
            return;
        }

        string? libraryPath = null;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            var candidate = Path.Combine(repoRoot, "libs", "native-osx-arm64", "libmlxsharp.dylib");
            if (File.Exists(candidate))
            {
                libraryPath = candidate;
            }
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            var candidate = Path.Combine(repoRoot, "libs", "native-libs", "libmlxsharp.so");
            if (File.Exists(candidate))
            {
                libraryPath = candidate;
            }
        }

        if (!string.IsNullOrWhiteSpace(libraryPath))
        {
            Environment.SetEnvironmentVariable("MLXSHARP_LIBRARY", libraryPath);

            var metalPath = Path.Combine(Path.GetDirectoryName(libraryPath)!, "mlx.metallib");
            if (File.Exists(metalPath))
            {
                Environment.SetEnvironmentVariable("MLX_METAL_PATH", metalPath);
            }
        }
    }

    private static void ConfigureModelPaths(string repoRoot)
    {
        var modelDir = Path.Combine(repoRoot, "model");
        if (Directory.Exists(modelDir))
        {
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MLXSHARP_MODEL_PATH")))
            {
                Environment.SetEnvironmentVariable("MLXSHARP_MODEL_PATH", modelDir);
            }
        }

        var tokenizerPath = Path.Combine(modelDir, "tokenizer.json");
        if (File.Exists(tokenizerPath) && string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MLXSHARP_TOKENIZER_PATH")))
        {
            Environment.SetEnvironmentVariable("MLXSHARP_TOKENIZER_PATH", tokenizerPath);
        }
    }
}

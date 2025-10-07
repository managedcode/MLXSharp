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
        var existing = Environment.GetEnvironmentVariable("MLXSHARP_LIBRARY");
        if (!string.IsNullOrWhiteSpace(existing) && File.Exists(existing))
        {
            ApplyNativeLibrary(existing);
            return;
        }

        string? libraryPath = null;
        if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
        {
            var candidates = new[]
            {
                Path.Combine(repoRoot, "libs", "native-osx-arm64", "libmlxsharp.dylib"),
                Path.Combine(repoRoot, "libs", "native-libs", "libmlxsharp.dylib"),
                Path.Combine(repoRoot, "libs", "native-libs", "osx-arm64", "libmlxsharp.dylib"),
            };

            libraryPath = Array.Find(candidates, File.Exists);
        }
        else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            var candidates = new[]
            {
                Path.Combine(repoRoot, "libs", "native-linux", "libmlxsharp.so"),
                Path.Combine(repoRoot, "libs", "native-libs", "libmlxsharp.so"),
                Path.Combine(repoRoot, "libs", "native-libs", "linux-x64", "libmlxsharp.so"),
            };

            libraryPath = Array.Find(candidates, File.Exists);
        }

        if (!string.IsNullOrWhiteSpace(libraryPath))
        {
            ApplyNativeLibrary(libraryPath);
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

    private static void ApplyNativeLibrary(string libraryPath)
    {
        Environment.SetEnvironmentVariable("MLXSHARP_LIBRARY", libraryPath);

        var metalPath = Path.Combine(Path.GetDirectoryName(libraryPath)!, "mlx.metallib");
        if (File.Exists(metalPath))
        {
            Environment.SetEnvironmentVariable("MLX_METAL_PATH", metalPath);
            Environment.SetEnvironmentVariable("MLX_METALLIB", metalPath);
        }

        var fileName = RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
            ? "libmlxsharp.dylib"
            : RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
                ? "libmlxsharp.so"
                : "libmlxsharp";

        TryCopy(libraryPath, Path.Combine(AppContext.BaseDirectory, fileName));
        if (File.Exists(metalPath))
        {
            TryCopy(metalPath, Path.Combine(AppContext.BaseDirectory, "mlx.metallib"));
        }
    }

    private static void TryCopy(string source, string destination)
    {
        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(destination)!);
            File.Copy(source, destination, overwrite: true);
        }
        catch
        {
            // best effort copy; ignore IO errors
        }
    }
}

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Text.Json;

namespace MLXSharp.Tests;

internal static class TestEnvironment
{
    private static int s_initialized;
    private static Exception? s_failure;

    public static void EnsureInitialized()
    {
        if (Interlocked.Exchange(ref s_initialized, 1) != 0)
        {
            if (s_failure is not null)
            {
                throw new InvalidOperationException("Failed to initialize MLXSharp test environment.", s_failure);
            }

            return;
        }

        try
        {
            var baseDirectory = AppContext.BaseDirectory;
            var repoRoot = ResolveRepoRoot(baseDirectory);

            EnsurePythonDependencies();
            ConfigureNativeLibrary(repoRoot);
            ConfigureModelPaths(repoRoot);
            s_failure = null;
        }
        catch (Exception ex)
        {
            s_failure = ex;
            throw new InvalidOperationException("Failed to initialize MLXSharp test environment.", ex);
        }
    }

    private static string ResolveRepoRoot(string baseDirectory)
    {
        var current = new DirectoryInfo(baseDirectory);
        while (current is not null)
        {
            var gitPath = Path.Combine(current.FullName, ".git");
            if (Directory.Exists(gitPath))
            {
                return current.FullName;
            }

            current = current.Parent;
        }

        throw new InvalidOperationException($"Unable to locate repository root starting from '{baseDirectory}'.");
    }

    private static void EnsurePythonDependencies()
    {
        const string script = """
import importlib.util
import subprocess
import sys

packages = {
    "mlx": "mlx",
    "mlx_lm": "mlx-lm",
    "huggingface_hub": "huggingface-hub",
    "sentencepiece": "sentencepiece",
    "tiktoken": "tiktoken",
}

missing = [pkg for module, pkg in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
""";

        RunPython(script, "Failed to ensure Python dependencies.");
    }

    private static void ConfigureNativeLibrary(string repoRoot)
    {
        var libraryPath = Environment.GetEnvironmentVariable("MLXSHARP_LIBRARY");
        if (TryValidateLibrary(libraryPath, out var resolvedLibrary, out var metallib))
        {
            ApplyNativeLibrary(resolvedLibrary, metallib);
            return;
        }

        foreach (var candidate in EnumerateLocalNativeCandidates(repoRoot))
        {
            if (TryValidateLibrary(candidate, out resolvedLibrary, out metallib))
            {
                ApplyNativeLibrary(resolvedLibrary, metallib);
                return;
            }
        }

        if (NativeBinaryManager.TryEnsureNativeLibrary(repoRoot, out resolvedLibrary, out var error) &&
            TryValidateLibrary(resolvedLibrary, out resolvedLibrary, out metallib))
        {
            ApplyNativeLibrary(resolvedLibrary, metallib);
            return;
        }

        var message = new StringBuilder();
        message.AppendLine("Unable to locate libmlxsharp native library.");
        if (!string.IsNullOrWhiteSpace(libraryPath))
        {
            message.AppendLine($"MLXSHARP_LIBRARY was set to '{libraryPath}' but the file was not found.");
        }
        if (!string.IsNullOrWhiteSpace(error))
        {
            message.AppendLine(error);
        }

        throw new InvalidOperationException(message.ToString());
    }

    private static bool TryValidateLibrary(string? libraryPath, out string resolvedLibrary, out string? metallib)
    {
        resolvedLibrary = string.Empty;
        metallib = null;

        if (string.IsNullOrWhiteSpace(libraryPath))
        {
            return false;
        }

        if (!File.Exists(libraryPath))
        {
            return false;
        }

        resolvedLibrary = Path.GetFullPath(libraryPath);

        var directory = Path.GetDirectoryName(resolvedLibrary)!;
        var metalCandidate = Path.Combine(directory, "mlx.metallib");
        if (File.Exists(metalCandidate))
        {
            metallib = metalCandidate;
        }

        return true;
    }

    private static IEnumerable<string> EnumerateLocalNativeCandidates(string repoRoot)
    {
        var libraryName = OperatingSystem.IsMacOS() ? "libmlxsharp.dylib" : "libmlxsharp.so";
        if (OperatingSystem.IsMacOS())
        {
            yield return Path.Combine(repoRoot, "native", "build", "macos", libraryName);
            yield return Path.Combine(repoRoot, "native", "build", "macos", "lib", libraryName);
            yield return Path.Combine(repoRoot, "libs", "native-osx-arm64", libraryName);
            yield return Path.Combine(repoRoot, "libs", "native-libs", "osx-arm64", libraryName);
        }
        else if (OperatingSystem.IsLinux())
        {
            yield return Path.Combine(repoRoot, "native", "build", "linux", libraryName);
            yield return Path.Combine(repoRoot, "libs", "native-linux", libraryName);
            yield return Path.Combine(repoRoot, "libs", "native-libs", "linux-x64", libraryName);
        }
    }

    private static void ApplyNativeLibrary(string libraryPath, string? metallibPath)
    {
        Environment.SetEnvironmentVariable("MLXSHARP_LIBRARY", libraryPath);

        if (!string.IsNullOrWhiteSpace(metallibPath))
        {
            Environment.SetEnvironmentVariable("MLX_METAL_PATH", metallibPath);
            Environment.SetEnvironmentVariable("MLX_METALLIB", metallibPath);
        }

        var fileName = RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
            ? "libmlxsharp.dylib"
            : RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
                ? "libmlxsharp.so"
                : "libmlxsharp";

        TryCopy(libraryPath, Path.Combine(AppContext.BaseDirectory, fileName));
        if (!string.IsNullOrWhiteSpace(metallibPath))
        {
            TryCopy(metallibPath!, Path.Combine(AppContext.BaseDirectory, "mlx.metallib"));
        }
    }

    private static void ConfigureModelPaths(string repoRoot)
    {
        var existingModel = Environment.GetEnvironmentVariable("MLXSHARP_MODEL_PATH");
        if (TryValidateModel(existingModel))
        {
            return;
        }

        var desiredModel = Environment.GetEnvironmentVariable("MLXSHARP_HF_MODEL_ID");
        if (string.IsNullOrWhiteSpace(desiredModel))
        {
            desiredModel = "mlx-community/Qwen1.5-0.5B-Chat-4bit";
        }

        var modelsRoot = Path.Combine(repoRoot, "models");
        Directory.CreateDirectory(modelsRoot);
        var targetDirectory = Path.Combine(modelsRoot, SanitizePath(desiredModel));

        if (!Directory.Exists(targetDirectory) || !TryValidateModel(targetDirectory))
        {
            DownloadModelSnapshot(desiredModel, targetDirectory);
        }

        if (!TryValidateModel(targetDirectory))
        {
            throw new InvalidOperationException($"Model '{desiredModel}' was not downloaded correctly to '{targetDirectory}'.");
        }

        Environment.SetEnvironmentVariable("MLXSHARP_MODEL_PATH", targetDirectory);

        var tokenizerPath = Path.Combine(targetDirectory, "tokenizer.json");
        if (!File.Exists(tokenizerPath))
        {
            throw new InvalidOperationException($"Model bundle at '{targetDirectory}' is missing tokenizer.json.");
        }

        Environment.SetEnvironmentVariable("MLXSHARP_TOKENIZER_PATH", tokenizerPath);
    }

    private static bool TryValidateModel(string? directory)
    {
        if (string.IsNullOrWhiteSpace(directory))
        {
            return false;
        }

        if (!Directory.Exists(directory))
        {
            return false;
        }

        var config = Path.Combine(directory, "config.json");
        var weights = Path.Combine(directory, "model.safetensors");
        return File.Exists(config) && (File.Exists(weights) || Directory.GetFiles(directory, "*.safetensors").Length > 0);
    }

    private static void DownloadModelSnapshot(string modelId, string destination)
    {
        Directory.CreateDirectory(destination);

        var token = Environment.GetEnvironmentVariable("HF_TOKEN");
        var includeToken = !string.IsNullOrWhiteSpace(token);
        var tokenLine = includeToken ? "kwargs[\"token\"] = os.environ.get(\"HF_TOKEN\")\n" : string.Empty;
        var modelLiteral = JsonSerializer.Serialize(modelId);
        var destinationLiteral = JsonSerializer.Serialize(destination);

        var script = $"""
import os
from huggingface_hub import snapshot_download

kwargs = dict(repo_id={modelLiteral}, local_dir={destinationLiteral}, local_dir_use_symlinks=False)
{tokenLine}snapshot_download(**kwargs)
""";

        RunPython(script, $"Failed to download Hugging Face model '{modelId}'.");
    }

    private static string SanitizePath(string value)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var builder = new StringBuilder(value.Length);
        foreach (var ch in value)
        {
            builder.Append(Array.IndexOf(invalid, ch) >= 0 ? '_' : ch);
        }

        return builder.ToString();
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
            // best effort
        }
    }

    private static void RunPython(string script, string errorMessage)
    {
        using var process = new Process
        {
            StartInfo = new ProcessStartInfo
            {
                FileName = "python3",
                ArgumentList = { "-" },
                RedirectStandardInput = true,
                RedirectStandardError = true,
                RedirectStandardOutput = true,
            }
        };

        process.Start();
        process.StandardInput.Write(script);
        process.StandardInput.Close();

        var stderr = process.StandardError.ReadToEnd();
        var stdout = process.StandardOutput.ReadToEnd();
        process.WaitForExit();

        if (process.ExitCode != 0)
        {
            var message = new StringBuilder(errorMessage);
            if (!string.IsNullOrWhiteSpace(stdout))
            {
                message.AppendLine();
                message.AppendLine("stdout:");
                message.AppendLine(stdout.Trim());
            }
            if (!string.IsNullOrWhiteSpace(stderr))
            {
                message.AppendLine();
                message.AppendLine("stderr:");
                message.AppendLine(stderr.Trim());
            }

            throw new InvalidOperationException(message.ToString());
        }
    }
}

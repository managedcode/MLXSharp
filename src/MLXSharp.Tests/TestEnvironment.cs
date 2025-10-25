using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Threading.Tasks;

namespace MLXSharp.Tests;

internal static class TestEnvironment
{
    private static readonly object s_sync = new();
    private static bool s_initialized;
    private static Exception? s_failure;
    private static readonly string[] s_requiredNativeExports =
    {
        "mlxsharp_create_session",
        "mlxsharp_generate_text",
        "mlxsharp_generate_embedding",
        "mlxsharp_generate_image",
        "mlxsharp_release_session",
        "mlxsharp_free_buffer",
        "mlxsharp_free_embedding",
    };

    public static IReadOnlyList<string> RequiredNativeExports => s_requiredNativeExports;

    public static void EnsureInitialized()
    {
        lock (s_sync)
        {
            if (s_initialized)
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
                var repoRoot = Path.GetFullPath(Path.Combine(baseDirectory, "..", "..", "..", ".."));

                var nativeAssets = EnsureNativeAssetsAsync(repoRoot).GetAwaiter().GetResult();
                ValidateNativeLibrary(nativeAssets.LibraryPath);
                ApplyNativeLibrary(nativeAssets.LibraryPath, nativeAssets.MetalLibraryPath);

                var modelAssets = EnsureModelAssetsAsync(repoRoot).GetAwaiter().GetResult();
                ConfigureModelEnvironment(modelAssets);

                s_initialized = true;
            }
            catch (Exception ex)
            {
                s_failure = ex;
                s_initialized = true;
                throw new InvalidOperationException("Failed to initialize MLXSharp test environment.", ex);
            }
        }
    }

    private static async Task<NativeAssets> EnsureNativeAssetsAsync(string repoRoot)
    {
        var existing = Environment.GetEnvironmentVariable("MLXSHARP_LIBRARY");
        if (!string.IsNullOrWhiteSpace(existing) && File.Exists(existing))
        {
            return new NativeAssets(existing, TryResolveMetallib(existing));
        }

        var runtime = DetermineRuntime();
        var libsRoot = Path.Combine(repoRoot, "libs", "native-libs", runtime.Rid);
        Directory.CreateDirectory(libsRoot);

        var libraryPath = Path.Combine(libsRoot, runtime.LibraryFileName);
        string? metallibPath = runtime.MetallibFileName is null
            ? null
            : Path.Combine(libsRoot, runtime.MetallibFileName);

        if (!File.Exists(libraryPath) || (metallibPath is not null && !File.Exists(metallibPath)))
        {
            await DownloadNativeReleaseAsync(libsRoot, runtime).ConfigureAwait(false);
        }

        if (!File.Exists(libraryPath))
        {
            throw new FileNotFoundException($"Native library '{libraryPath}' was not downloaded.");
        }

        if (metallibPath is not null && !File.Exists(metallibPath))
        {
            throw new FileNotFoundException($"Metal kernels '{metallibPath}' were not downloaded.");
        }

        return new NativeAssets(libraryPath, metallibPath);
    }

    private static async Task DownloadNativeReleaseAsync(string destinationRoot, RuntimeInfo runtime)
    {
        var repo = Environment.GetEnvironmentVariable("MLXSHARP_NATIVE_REPO");
        if (string.IsNullOrWhiteSpace(repo))
        {
            repo = "ManagedCode/MLXSharp";
        }

        var tag = Environment.GetEnvironmentVariable("MLXSHARP_NATIVE_TAG");
        var releaseEndpoint = string.IsNullOrWhiteSpace(tag)
            ? $"https://api.github.com/repos/{repo}/releases/latest"
            : $"https://api.github.com/repos/{repo}/releases/tags/{tag}";

        using var client = CreateHttpClient();

        using var metadataResponse = await client.GetAsync(releaseEndpoint).ConfigureAwait(false);
        metadataResponse.EnsureSuccessStatusCode();
        await using var metadataStream = await metadataResponse.Content.ReadAsStreamAsync().ConfigureAwait(false);
        using var document = await JsonDocument.ParseAsync(metadataStream).ConfigureAwait(false);

        if (!document.RootElement.TryGetProperty("assets", out var assetsElement))
        {
            throw new InvalidOperationException($"Release metadata for '{repo}' did not include any assets.");
        }

        string? assetUrl = null;
        string? assetName = null;
        foreach (var asset in assetsElement.EnumerateArray())
        {
            if (!asset.TryGetProperty("name", out var nameElement))
            {
                continue;
            }

            var name = nameElement.GetString();
            if (string.IsNullOrWhiteSpace(name))
            {
                continue;
            }

            if (!name.StartsWith("ManagedCode.MLXSharp.", StringComparison.OrdinalIgnoreCase) ||
                !name.EndsWith(".nupkg", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            if (!asset.TryGetProperty("url", out var urlElement))
            {
                continue;
            }

            var url = urlElement.GetString();
            if (string.IsNullOrWhiteSpace(url))
            {
                continue;
            }

            assetUrl = url;
            assetName = name;
            break;
        }

        if (string.IsNullOrWhiteSpace(assetUrl) || string.IsNullOrWhiteSpace(assetName))
        {
            throw new InvalidOperationException($"Unable to locate ManagedCode.MLXSharp nupkg asset for repository '{repo}'.");
        }

        var tempFile = Path.Combine(Path.GetTempPath(), $"mlxsharp-native-{Guid.NewGuid():N}.nupkg");

        try
        {
            using (var request = new HttpRequestMessage(HttpMethod.Get, assetUrl))
            {
                request.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/octet-stream"));

                using var assetResponse = await client.SendAsync(request).ConfigureAwait(false);
                assetResponse.EnsureSuccessStatusCode();

                await using var downloadStream = await assetResponse.Content.ReadAsStreamAsync().ConfigureAwait(false);
                await using var fileStream = File.Create(tempFile);
                await downloadStream.CopyToAsync(fileStream).ConfigureAwait(false);
            }

            using var archiveStream = File.OpenRead(tempFile);
            using var archive = new ZipArchive(archiveStream, ZipArchiveMode.Read);

            var prefix = $"runtimes/{runtime.Rid}/native/";
            var extractedAny = false;

            foreach (var entry in archive.Entries)
            {
                var normalized = entry.FullName.Replace('\\', '/');
                if (!normalized.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }

                if (normalized.EndsWith('/'))
                {
                    continue;
                }

                var relative = normalized.Substring(prefix.Length);
                if (string.IsNullOrWhiteSpace(relative))
                {
                    continue;
                }

                var destinationPath = Path.Combine(destinationRoot, relative);
                Directory.CreateDirectory(Path.GetDirectoryName(destinationPath)!);
                entry.ExtractToFile(destinationPath, overwrite: true);
                extractedAny = true;
            }

            if (!extractedAny)
            {
                throw new InvalidOperationException($"Native runtime '{runtime.Rid}' was not present in asset '{assetName}'.");
            }
        }
        finally
        {
            TryDelete(tempFile);
        }
    }

    private static async Task<ModelAssets> EnsureModelAssetsAsync(string repoRoot)
    {
        var modelPath = Environment.GetEnvironmentVariable("MLXSHARP_MODEL_PATH");
        if (string.IsNullOrWhiteSpace(modelPath))
        {
            modelPath = Path.Combine(repoRoot, "model");
        }

        Directory.CreateDirectory(modelPath);

        var tokenizerPath = Path.Combine(modelPath, "tokenizer.json");
        var hasWeights = Directory.EnumerateFiles(modelPath, "*.safetensors", SearchOption.TopDirectoryOnly).Any();

        if (!File.Exists(tokenizerPath) || !hasWeights)
        {
            await DownloadModelAsync(modelPath).ConfigureAwait(false);
            hasWeights = Directory.EnumerateFiles(modelPath, "*.safetensors", SearchOption.TopDirectoryOnly).Any();
        }

        if (!File.Exists(tokenizerPath))
        {
            throw new FileNotFoundException($"Tokenizer not found in '{modelPath}'.");
        }

        if (!hasWeights)
        {
            throw new FileNotFoundException($"Model weights (*.safetensors) were not downloaded to '{modelPath}'.");
        }

        return new ModelAssets(modelPath, tokenizerPath);
    }

    private static async Task DownloadModelAsync(string modelDirectory)
    {
        var modelId = Environment.GetEnvironmentVariable("MLXSHARP_HF_MODEL_ID");
        if (string.IsNullOrWhiteSpace(modelId))
        {
            modelId = "mlx-community/Qwen1.5-0.5B-Chat-4bit";
        }

        var python = FindPythonExecutable();
        if (python is null)
        {
            throw new InvalidOperationException("Python 3 is required to download MLX model assets but was not found on PATH.");
        }

        await EnsurePythonPackagesAsync(python).ConfigureAwait(false);

        var scriptPath = Path.Combine(Path.GetTempPath(), $"mlx_download_{Guid.NewGuid():N}.py");
        var script = """
import os
from huggingface_hub import snapshot_download

model_id = os.environ["MLXSHARP_HF_MODEL_ID"]
model_dir = os.environ["MLXSHARP_MODEL_PATH"]
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None

snapshot_download(
    repo_id=model_id,
    local_dir=model_dir,
    local_dir_use_symlinks=False,
    token=token,
    resume_download=True,
)
""";

        await File.WriteAllTextAsync(scriptPath, script).ConfigureAwait(false);

        try
        {
            var environment = new Dictionary<string, string?>
            {
                ["MLXSHARP_MODEL_PATH"] = modelDirectory,
                ["MLXSHARP_HF_MODEL_ID"] = modelId,
            };

            var hfToken = Environment.GetEnvironmentVariable("HF_TOKEN") ?? Environment.GetEnvironmentVariable("HUGGINGFACE_TOKEN");
            if (!string.IsNullOrWhiteSpace(hfToken))
            {
                environment["HF_TOKEN"] = hfToken;
            }

            var result = await RunProcessAsync(python, new[] { scriptPath }, environment).ConfigureAwait(false);
            if (result.ExitCode != 0)
            {
                throw new InvalidOperationException($"Hugging Face snapshot download failed with exit code {result.ExitCode}:{Environment.NewLine}{result.StandardError}");
            }
        }
        finally
        {
            TryDelete(scriptPath);
        }
    }

    private static async Task EnsurePythonPackagesAsync(string python)
    {
        var args = new[] { "-m", "pip", "install", "--quiet", "--upgrade", "huggingface_hub", "mlx", "mlx-lm" };
        var result = await RunProcessAsync(python, args).ConfigureAwait(false);
        if (result.ExitCode != 0)
        {
            throw new InvalidOperationException($"Failed to install required Python packages (exit code {result.ExitCode}).{Environment.NewLine}{result.StandardError}");
        }
    }

    private static HttpClient CreateHttpClient()
    {
        var client = new HttpClient();
        client.DefaultRequestHeaders.UserAgent.ParseAdd("MLXSharp.Tests/1.0");

        var token = Environment.GetEnvironmentVariable("GITHUB_TOKEN") ?? Environment.GetEnvironmentVariable("GH_TOKEN");
        if (!string.IsNullOrWhiteSpace(token))
        {
            client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", token);
        }

        return client;
    }

    private static RuntimeInfo DetermineRuntime()
    {
        if (OperatingSystem.IsMacOS())
        {
            if (RuntimeInformation.ProcessArchitecture != Architecture.Arm64)
            {
                throw new PlatformNotSupportedException("macOS builds require arm64 native libraries.");
            }

            return new RuntimeInfo("osx-arm64", "libmlxsharp.dylib", "mlx.metallib");
        }

        if (OperatingSystem.IsLinux())
        {
            if (RuntimeInformation.ProcessArchitecture != Architecture.X64)
            {
                throw new PlatformNotSupportedException("Linux builds require x64 native libraries.");
            }

            return new RuntimeInfo("linux-x64", "libmlxsharp.so", null);
        }

        throw new PlatformNotSupportedException("Only Linux x64 and macOS arm64 are supported by the native MLXSharp tests.");
    }

    private static void ValidateNativeLibrary(string path)
    {
        if (!NativeLibrary.TryLoad(path, out var handle))
        {
            throw new InvalidOperationException($"Unable to load native library from '{path}'.");
        }

        try
        {
            foreach (var export in s_requiredNativeExports)
            {
                if (!NativeLibrary.TryGetExport(handle, export, out _))
                {
                    throw new InvalidOperationException($"Native library at '{path}' is missing required export '{export}'.");
                }
            }
        }
        finally
        {
            NativeLibrary.Free(handle);
        }
    }

    private static void ApplyNativeLibrary(string libraryPath, string? metallibPath)
    {
        Environment.SetEnvironmentVariable("MLXSHARP_LIBRARY", libraryPath);

        if (!string.IsNullOrWhiteSpace(metallibPath) && File.Exists(metallibPath))
        {
            Environment.SetEnvironmentVariable("MLX_METAL_PATH", metallibPath);
            Environment.SetEnvironmentVariable("MLX_METALLIB", metallibPath);
        }

        var fileName = OperatingSystem.IsMacOS()
            ? "libmlxsharp.dylib"
            : OperatingSystem.IsLinux()
                ? "libmlxsharp.so"
                : "mlxsharp.dll";

        TryCopy(libraryPath, Path.Combine(AppContext.BaseDirectory, fileName));

        if (!string.IsNullOrWhiteSpace(metallibPath) && File.Exists(metallibPath))
        {
            TryCopy(metallibPath, Path.Combine(AppContext.BaseDirectory, "mlx.metallib"));
        }
    }

    private static void ConfigureModelEnvironment(ModelAssets assets)
    {
        Environment.SetEnvironmentVariable("MLXSHARP_MODEL_PATH", assets.ModelDirectory);
        Environment.SetEnvironmentVariable("MLXSHARP_TOKENIZER_PATH", assets.TokenizerPath);
    }

    private static string? TryResolveMetallib(string libraryPath)
    {
        var directory = Path.GetDirectoryName(libraryPath);
        if (string.IsNullOrWhiteSpace(directory))
        {
            return null;
        }

        var candidate = Path.Combine(directory, "mlx.metallib");
        return File.Exists(candidate) ? candidate : null;
    }

    private static string? FindPythonExecutable()
    {
        foreach (var candidate in EnumeratePythonCandidates())
        {
            var resolved = ResolveExecutable(candidate);
            if (resolved is not null)
            {
                return resolved;
            }
        }

        return null;
    }

    private static IEnumerable<string> EnumeratePythonCandidates()
    {
        var configured = Environment.GetEnvironmentVariable("PYTHON");
        if (!string.IsNullOrWhiteSpace(configured))
        {
            yield return configured;
        }

        yield return "python3";
        yield return "python";
    }

    private static string? ResolveExecutable(string command)
    {
        if (Path.IsPathRooted(command) && File.Exists(command))
        {
            return command;
        }

        var suffixes = OperatingSystem.IsWindows()
            ? new[] { ".exe", ".bat", string.Empty }
            : new[] { string.Empty };

        var pathVariable = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
        foreach (var segment in pathVariable.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            var trimmed = segment.Trim();
            if (string.IsNullOrEmpty(trimmed))
            {
                continue;
            }

            foreach (var suffix in suffixes)
            {
                var candidate = Path.Combine(trimmed, command + suffix);
                if (File.Exists(candidate))
                {
                    return candidate;
                }
            }
        }

        return null;
    }

    private static async Task<ProcessResult> RunProcessAsync(string fileName, IEnumerable<string> arguments, IDictionary<string, string?>? environment = null)
    {
        var psi = new ProcessStartInfo
        {
            FileName = fileName,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
            UseShellExecute = false,
        };

        foreach (var argument in arguments)
        {
            psi.ArgumentList.Add(argument);
        }

        if (environment is not null)
        {
            foreach (var pair in environment)
            {
                if (pair.Value is null)
                {
                    continue;
                }

                psi.Environment[pair.Key] = pair.Value;
            }
        }

        using var process = Process.Start(psi) ?? throw new InvalidOperationException($"Failed to start process '{fileName}'.");

        var stdoutTask = process.StandardOutput.ReadToEndAsync();
        var stderrTask = process.StandardError.ReadToEndAsync();

        await Task.WhenAll(process.WaitForExitAsync(), stdoutTask, stderrTask).ConfigureAwait(false);

        return new ProcessResult(process.ExitCode, stdoutTask.Result, stderrTask.Result);
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
            // best effort copy
        }
    }

    private static void TryDelete(string path)
    {
        try
        {
            if (!string.IsNullOrWhiteSpace(path) && File.Exists(path))
            {
                File.Delete(path);
            }
        }
        catch
        {
            // ignore cleanup failures
        }
    }

    private sealed record NativeAssets(string LibraryPath, string? MetalLibraryPath);

    private sealed record ModelAssets(string ModelDirectory, string TokenizerPath);

    private sealed record RuntimeInfo(string Rid, string LibraryFileName, string? MetallibFileName);

    private readonly record struct ProcessResult(int ExitCode, string StandardOutput, string StandardError);
}

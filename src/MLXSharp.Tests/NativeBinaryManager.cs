using System;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Text.Json;

namespace MLXSharp.Tests;

internal static class NativeBinaryManager
{
    private const string PackageId = "managedcode.mlxsharp";
    private const string BaseUrl = "https://api.nuget.org/v3-flatcontainer";

    private static readonly object s_sync = new();
    private static bool s_attempted;
    private static string? s_cachedPath;
    private static string? s_lastError;

    public static bool TryEnsureNativeLibrary(string repoRoot, out string? libraryPath, out string? error)
    {
        if (!OperatingSystem.IsMacOS() && !OperatingSystem.IsLinux())
        {
            libraryPath = null;
            error = "Official native binaries are only published for macOS and Linux.";
            return false;
        }

        lock (s_sync)
        {
            if (!string.IsNullOrEmpty(s_cachedPath) && File.Exists(s_cachedPath))
            {
                libraryPath = s_cachedPath;
                error = null;
                return true;
            }

            if (s_attempted)
            {
                libraryPath = s_cachedPath;
                error = s_lastError;
                return libraryPath is not null;
            }

            s_attempted = true;

            try
            {
                var path = DownloadOfficialBinary(repoRoot);
                s_cachedPath = path;
                s_lastError = null;
                libraryPath = path;
                error = null;
                return true;
            }
            catch (Exception ex)
            {
                s_cachedPath = null;
                s_lastError = ex.Message;
                libraryPath = null;
                error = s_lastError;
                return false;
            }
        }
    }

    private static string DownloadOfficialBinary(string repoRoot)
    {
        var rid = GetRuntimeIdentifier();
        var fileName = OperatingSystem.IsMacOS() ? "libmlxsharp.dylib" : "libmlxsharp.so";
        var nativeDirectory = Path.Combine(repoRoot, "libs", "native-libs", rid);
        Directory.CreateDirectory(nativeDirectory);

        var destination = Path.Combine(nativeDirectory, fileName);
        if (File.Exists(destination))
        {
            return destination;
        }

        using var client = new HttpClient();
        var version = ResolvePackageVersion(client);
        var packageUrl = $"{BaseUrl}/{PackageId}/{version}/{PackageId}.{version}.nupkg";

        using var packageStream = client.GetStreamAsync(packageUrl).GetAwaiter().GetResult();
        var tempFile = Path.GetTempFileName();
        try
        {
            using (var fileStream = File.OpenWrite(tempFile))
            {
                packageStream.CopyTo(fileStream);
            }

            using var archive = ZipFile.OpenRead(tempFile);
            var entryPath = $"runtimes/{rid}/native/{fileName}";
            var entry = archive.GetEntry(entryPath) ??
                        throw new InvalidOperationException($"The official package does not contain {entryPath}.");

            entry.ExtractToFile(destination, overwrite: true);
            return destination;
        }
        finally
        {
            try
            {
                File.Delete(tempFile);
            }
            catch
            {
                // ignore cleanup errors
            }
        }
    }

    private static string ResolvePackageVersion(HttpClient client)
    {
        var overrideVersion = Environment.GetEnvironmentVariable("MLXSHARP_OFFICIAL_NATIVE_VERSION");
        if (!string.IsNullOrWhiteSpace(overrideVersion))
        {
            return overrideVersion.Trim();
        }

        var indexUrl = $"{BaseUrl}/{PackageId}/index.json";
        using var stream = client.GetStreamAsync(indexUrl).GetAwaiter().GetResult();
        using var document = JsonDocument.Parse(stream);
        if (!document.RootElement.TryGetProperty("versions", out var versions) || versions.GetArrayLength() == 0)
        {
            throw new InvalidOperationException("Unable to determine the latest ManagedCode.MLXSharp package version.");
        }

        return versions[versions.GetArrayLength() - 1].GetString()
            ?? throw new InvalidOperationException("ManagedCode.MLXSharp package version entry was null.");
    }

    private static string GetRuntimeIdentifier()
    {
        if (OperatingSystem.IsMacOS())
        {
            return "osx-arm64";
        }

        if (OperatingSystem.IsLinux())
        {
            return "linux-x64";
        }

        throw new PlatformNotSupportedException("Unsupported platform for native MLXSharp binaries.");
    }
}

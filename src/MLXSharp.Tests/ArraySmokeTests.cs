using System;
using System.IO;
using System.Runtime.InteropServices;
using Xunit;

namespace MLXSharp.Tests;

public sealed class NativeLibrarySmokeTests
{
    [Fact]
    public void NativeLibraryProvidesExpectedExports()
    {
        TestEnvironment.EnsureInitialized();

        var libraryPath = Environment.GetEnvironmentVariable("MLXSHARP_LIBRARY");
        Assert.False(string.IsNullOrWhiteSpace(libraryPath));
        Assert.True(File.Exists(libraryPath));

        if (!NativeLibrary.TryLoad(libraryPath!, out var handle))
        {
            throw new InvalidOperationException($"Unable to load native library from '{libraryPath}'.");
        }

        try
        {
            foreach (var export in TestEnvironment.RequiredNativeExports)
            {
                Assert.True(
                    NativeLibrary.TryGetExport(handle, export, out _),
                    $"Native library at '{libraryPath}' is missing required export '{export}'.");
            }
        }
        finally
        {
            NativeLibrary.Free(handle);
        }
    }
}

using System;
using System.Runtime.InteropServices;

namespace MLXSharp.Native;

internal static partial class MlxNativeMethods
{
    [LibraryImport("libmlxsharp", EntryPoint = "mlxsharp_create_session", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int CreateSession(string chatModelId, string embeddingModelId, string imageModelId, out SafeMlxSessionHandle session);

    [LibraryImport("libmlxsharp", EntryPoint = "mlxsharp_release_session")]
    public static partial void ReleaseSession(nint session);

    [LibraryImport("libmlxsharp", EntryPoint = "mlxsharp_generate_text", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int GenerateText(SafeMlxSessionHandle session, string prompt, out MlxNativeStringHandle response, out MlxUsage usage);

    [LibraryImport("libmlxsharp", EntryPoint = "mlxsharp_generate_embedding", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int GenerateEmbedding(SafeMlxSessionHandle session, string text, out nint embeddingPointer, out int dimension, out MlxUsage usage);

    [LibraryImport("libmlxsharp", EntryPoint = "mlxsharp_free_embedding")]
    public static partial void FreeEmbedding(nint embeddingPointer);

    [LibraryImport("libmlxsharp", EntryPoint = "mlxsharp_generate_image", StringMarshalling = StringMarshalling.Utf8)]
    public static partial int GenerateImage(SafeMlxSessionHandle session, string prompt, int width, int height, out nint bufferPointer, out int length, out MlxUsage usage);

    [LibraryImport("libmlxsharp", EntryPoint = "mlxsharp_free_buffer")]
    public static partial void FreeBuffer(nint bufferPointer);
}

[StructLayout(LayoutKind.Sequential)]
internal struct MlxUsage
{
    public int InputTokens;
    public int OutputTokens;
}

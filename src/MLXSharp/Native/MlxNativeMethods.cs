using System;
using System.Runtime.InteropServices;
using System.Text;
using MLXSharp.Core;

namespace MLXSharp.Native;

internal enum MlxStatus : int
{
    Success = 0,
    InvalidArgument = -1,
    OutOfMemory = -2,
    DeviceUnavailable = -3,
    RuntimeError = -4,
}

internal static partial class MlxNativeMethods
{
    private const string LibraryName = "libmlxsharp";

    public static string GetLastErrorString()
    {
        Span<byte> initial = stackalloc byte[256];
        var written = GetLastError(initial, (nuint)initial.Length);
        if (written < initial.Length - 1)
        {
            return Encoding.UTF8.GetString(initial[..written]);
        }

        byte[] buffer = new byte[written];
        written = GetLastError(buffer, (nuint)buffer.Length);
        return Encoding.UTF8.GetString(buffer[..written]);
    }

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_get_last_error")]
    private static partial int GetLastError(Span<byte> buffer, nuint length);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_get_last_error")]
    private static partial int GetLastError(byte[] buffer, nuint length);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_context_create")]
    public static partial MlxStatus ContextCreate(MlxDeviceKind kind, int deviceIndex, out SafeMlxContextHandle context);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_context_retain")]
    public static partial void ContextRetain(SafeMlxContextHandle context);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_context_release")]
    public static partial void ContextRelease(nint context);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_array_from_buffer")]
    public static unsafe partial MlxStatus ArrayFromBuffer(
        SafeMlxContextHandle context,
        void* data,
        nuint elementCount,
        long* shape,
        int rank,
        MlxDType dtype,
        out SafeMlxArrayHandle array);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_array_zeros")]
    public static unsafe partial MlxStatus ArrayZeros(
        SafeMlxContextHandle context,
        long* shape,
        int rank,
        MlxDType dtype,
        out SafeMlxArrayHandle array);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_array_retain")]
    public static partial void ArrayRetain(SafeMlxArrayHandle array);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_array_release")]
    public static partial void ArrayRelease(nint array);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_array_get_shape")]
    public static unsafe partial MlxStatus ArrayGetShape(
        SafeMlxArrayHandle array,
        long* shape,
        int maxRank,
        out int actualRank);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_array_get_dtype")]
    public static partial MlxDType ArrayGetDType(SafeMlxArrayHandle array);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_array_get_size")]
    public static partial nuint ArrayGetSize(SafeMlxArrayHandle array);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_array_copy_to_buffer")]
    public static unsafe partial MlxStatus ArrayCopyToBuffer(
        SafeMlxArrayHandle array,
        void* destination,
        nuint elementCount);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_array_add")]
    public static partial MlxStatus ArrayAdd(
        SafeMlxArrayHandle left,
        SafeMlxArrayHandle right,
        out SafeMlxArrayHandle result);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_array_subtract")]
    public static partial MlxStatus ArraySubtract(
        SafeMlxArrayHandle left,
        SafeMlxArrayHandle right,
        out SafeMlxArrayHandle result);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_array_multiply")]
    public static partial MlxStatus ArrayMultiply(
        SafeMlxArrayHandle left,
        SafeMlxArrayHandle right,
        out SafeMlxArrayHandle result);

    [LibraryImport(LibraryName, EntryPoint = "mlxsharp_array_divide")]
    public static partial MlxStatus ArrayDivide(
        SafeMlxArrayHandle left,
        SafeMlxArrayHandle right,
        out SafeMlxArrayHandle result);
}

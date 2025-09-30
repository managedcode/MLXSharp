using System;
using System.Runtime.InteropServices;

namespace MLXSharp.Native;

internal sealed class SafeMlxSessionHandle : SafeHandle
{
    public SafeMlxSessionHandle()
        : base(nint.Zero, ownsHandle: true)
    {
    }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        if (!IsInvalid)
        {
            MlxNativeMethods.ReleaseSession(handle);
        }

        return true;
    }
}

internal sealed class MlxNativeStringHandle : SafeHandle
{
    public MlxNativeStringHandle()
        : base(nint.Zero, ownsHandle: true)
    {
    }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        if (!IsInvalid)
        {
            MlxNativeMethods.FreeBuffer(handle);
        }

        return true;
    }

    public string GetString() => IsInvalid ? string.Empty : Marshal.PtrToStringUTF8(handle) ?? string.Empty;
}

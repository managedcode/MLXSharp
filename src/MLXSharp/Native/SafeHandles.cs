using System;
using System.Runtime.InteropServices;

namespace MLXSharp.Native;

internal sealed class SafeMlxContextHandle : SafeHandle
{
    public SafeMlxContextHandle()
        : base(nint.Zero, ownsHandle: true)
    {
    }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        if (!IsInvalid)
        {
            MlxNativeMethods.ContextRelease(handle);
        }

        return true;
    }
}

internal sealed class SafeMlxArrayHandle : SafeHandle
{
    public SafeMlxArrayHandle()
        : base(nint.Zero, ownsHandle: true)
    {
    }

    public override bool IsInvalid => handle == nint.Zero;

    protected override bool ReleaseHandle()
    {
        if (!IsInvalid)
        {
            MlxNativeMethods.ArrayRelease(handle);
        }

        return true;
    }
}


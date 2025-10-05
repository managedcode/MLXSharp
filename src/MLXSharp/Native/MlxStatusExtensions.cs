namespace MLXSharp.Native;

internal static class MlxStatusExtensions
{
    public static void ThrowIfFailed(this MlxStatus status)
    {
        if (status != MlxStatus.Success)
        {
            var message = MlxNativeMethods.GetLastErrorString();
            throw new MlxNativeException(status, message);
        }
    }
}


using System;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using MLXSharp.Native;

namespace MLXSharp.Core;

/// <summary>
/// Managed projection of an MLX array handle.
/// </summary>
public sealed class MlxArray : IDisposable
{
    private bool _disposed;

    private MlxArray(SafeMlxArrayHandle handle)
    {
        Handle = handle;
    }

    internal SafeMlxArrayHandle Handle { get; }

    public MlxDType DType => EnsureNotDisposed(() => MlxNativeMethods.ArrayGetDType(Handle));

    public nuint Size => EnsureNotDisposed(() => MlxNativeMethods.ArrayGetSize(Handle));

    public int Rank
    {
        get
        {
            EnsureNotDisposed();
            unsafe
            {
                var status = MlxNativeMethods.ArrayGetShape(Handle, null, 0, out var rank);
                status.ThrowIfFailed();
                return rank;
            }
        }
    }

    public long[] Shape
    {
        get
        {
            EnsureNotDisposed();
            var rank = Rank;
            if (rank == 0)
            {
                return Array.Empty<long>();
            }

            var shape = new long[rank];
            unsafe
            {
                fixed (long* pShape = shape)
                {
                    var status = MlxNativeMethods.ArrayGetShape(Handle, pShape, rank, out var actualRank);
                    status.ThrowIfFailed();
                    if (actualRank != rank)
                    {
                        return shape.Take(actualRank).ToArray();
                    }
                }
            }

            return shape;
        }
    }

    public static MlxArray From(MlxContext context, ReadOnlySpan<float> data, ReadOnlySpan<long> shape)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (shape.IsEmpty)
        {
            throw new ArgumentException("Shape must contain at least one dimension.", nameof(shape));
        }

        var expected = ComputeElementCount(shape);
        if ((nuint)data.Length != expected)
        {
            throw new ArgumentException($"Data length {data.Length} does not match shape product {expected}.", nameof(data));
        }

        unsafe
        {
            fixed (float* pData = data)
            fixed (long* pShape = shape)
            {
                var status = MlxNativeMethods.ArrayFromBuffer(
                    context.Handle,
                    pData,
                    (nuint)data.Length,
                    pShape,
                    shape.Length,
                    MlxDType.Float32,
                    out var handle);

                status.ThrowIfFailed();
                return new MlxArray(handle);
            }
        }
    }

    public static MlxArray Zeros(MlxContext context, ReadOnlySpan<long> shape, MlxDType dtype = MlxDType.Float32)
    {
        ArgumentNullException.ThrowIfNull(context);
        if (shape.IsEmpty)
        {
            throw new ArgumentException("Shape must contain at least one dimension.", nameof(shape));
        }

        unsafe
        {
            fixed (long* pShape = shape)
            {
                var status = MlxNativeMethods.ArrayZeros(
                    context.Handle,
                    pShape,
                    shape.Length,
                    dtype,
                    out var handle);

                status.ThrowIfFailed();
                return new MlxArray(handle);
            }
        }
    }

    public float[] ToArrayFloat32()
    {
        EnsureNotDisposed();
        if (DType != MlxDType.Float32)
        {
            throw new InvalidOperationException($"Array dtype {DType} does not match float32.");
        }

        var length = (int)Size;
        var buffer = new float[length];
        CopyTo(buffer);
        return buffer;
    }

    public void CopyTo(Span<float> destination)
    {
        EnsureNotDisposed();
        if (DType != MlxDType.Float32)
        {
            throw new InvalidOperationException($"Array dtype {DType} does not match float32.");
        }

        if ((nuint)destination.Length < Size)
        {
            throw new ArgumentException("Destination span is too small for array contents.", nameof(destination));
        }

        unsafe
        {
            fixed (float* pDest = destination)
            {
                var status = MlxNativeMethods.ArrayCopyToBuffer(Handle, pDest, (nuint)destination.Length);
                status.ThrowIfFailed();
            }
        }
    }

    public static MlxArray Add(MlxArray left, MlxArray right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        var status = MlxNativeMethods.ArrayAdd(left.Handle, right.Handle, out var handle);
        status.ThrowIfFailed();
        return new MlxArray(handle);
    }

    public static MlxArray Subtract(MlxArray left, MlxArray right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        var status = MlxNativeMethods.ArraySubtract(left.Handle, right.Handle, out var handle);
        status.ThrowIfFailed();
        return new MlxArray(handle);
    }

    public static MlxArray Multiply(MlxArray left, MlxArray right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        var status = MlxNativeMethods.ArrayMultiply(left.Handle, right.Handle, out var handle);
        status.ThrowIfFailed();
        return new MlxArray(handle);
    }

    public static MlxArray Divide(MlxArray left, MlxArray right)
    {
        ArgumentNullException.ThrowIfNull(left);
        ArgumentNullException.ThrowIfNull(right);
        var status = MlxNativeMethods.ArrayDivide(left.Handle, right.Handle, out var handle);
        status.ThrowIfFailed();
        return new MlxArray(handle);
    }

    public MlxArray Add(MlxArray other) => Add(this, other);

    public MlxArray Subtract(MlxArray other) => Subtract(this, other);

    public MlxArray Multiply(MlxArray other) => Multiply(this, other);

    public MlxArray Divide(MlxArray other) => Divide(this, other);

    public void Dispose()
    {
        if (_disposed)
        {
            return;
        }

        Handle.Dispose();
        _disposed = true;
    }

    private static nuint ComputeElementCount(ReadOnlySpan<long> shape)
    {
        nuint result = 1;
        foreach (var dim in shape)
        {
            if (dim <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(shape), dim, "Shape dimensions must be positive.");
            }

            result *= (nuint)dim;
        }

        return result;
    }

    private void EnsureNotDisposed()
    {
        if (_disposed)
        {
            throw new ObjectDisposedException(nameof(MlxArray));
        }
    }

    private T EnsureNotDisposed<T>(Func<T> accessor)
    {
        EnsureNotDisposed();
        return accessor();
    }
}


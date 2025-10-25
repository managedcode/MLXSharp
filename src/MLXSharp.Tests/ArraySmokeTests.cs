using System;
using MLXSharp.Core;
using Xunit;

namespace MLXSharp.Tests;

public sealed class ArraySmokeTests
{
    [Fact]
    public void AddTwoFloatArrays()
    {
        TestEnvironment.EnsureInitialized();

        using var context = MlxContext.CreateCpu();

        ReadOnlySpan<float> leftData = stackalloc float[] { 1f, 2f, 3f, 4f };
        ReadOnlySpan<float> rightData = stackalloc float[] { 5f, 6f, 7f, 8f };
        ReadOnlySpan<long> shape = stackalloc long[] { 2, 2 };

        using var left = MlxArray.From(context, leftData, shape);
        using var right = MlxArray.From(context, rightData, shape);
        using var result = MlxArray.Add(left, right);

        Assert.Equal(new[] { 6f, 8f, 10f, 12f }, result.ToArrayFloat32());
        Assert.Equal(shape.ToArray(), result.Shape);
        Assert.Equal(MlxDType.Float32, result.DType);
    }

    [Fact]
    public void ZerosAllocatesRequestedShape()
    {
        TestEnvironment.EnsureInitialized();

        using var context = MlxContext.CreateCpu();
        ReadOnlySpan<long> shape = stackalloc long[] { 3, 1 };

        using var zeros = MlxArray.Zeros(context, shape, MlxDType.Float32);

        Assert.Equal(MlxDType.Float32, zeros.DType);
        Assert.Equal(shape.ToArray(), zeros.Shape);
        Assert.All(zeros.ToArrayFloat32(), value => Assert.Equal(0f, value));
    }
}

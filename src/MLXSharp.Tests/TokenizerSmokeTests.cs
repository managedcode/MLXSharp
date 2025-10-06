using System;
using System.IO;
using MLXSharp.Tokenization;
using Xunit;

namespace MLXSharp.Tests;

public sealed class TokenizerSmokeTests
{
    [RequiresTokenizerFact]
    public void TokenizerRoundTrip()
    {
        var path = Environment.GetEnvironmentVariable("MLXSHARP_TOKENIZER_PATH");
        Assert.False(string.IsNullOrWhiteSpace(path));

        var tokenizer = MlxTokenizer.Load(path!);
        var encoding = tokenizer.Encode("Hello MLX");
        Assert.NotNull(encoding.Tokens);
        Assert.NotEmpty(encoding.Tokens);

        var decoded = tokenizer.Decode(encoding.Tokens);
        Assert.False(string.IsNullOrWhiteSpace(decoded));
    }
}

internal sealed class RequiresTokenizerFactAttribute : FactAttribute
{
    public RequiresTokenizerFactAttribute()
    {
        TestEnvironment.EnsureInitialized();
        var tokenizerPath = Environment.GetEnvironmentVariable("MLXSHARP_TOKENIZER_PATH");
        if (string.IsNullOrWhiteSpace(tokenizerPath) || !File.Exists(tokenizerPath))
        {
            Skip = "MLXSHARP_TOKENIZER_PATH is not set or does not point to an existing tokenizer.json.";
        }
    }
}

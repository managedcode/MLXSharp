using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using Microsoft.CSharp.RuntimeBinder;
using Microsoft.ML.Tokenizers;

namespace MLXSharp.Tokenization;

/// <summary>
/// Thin wrapper around <see cref="Tokenizer"/> that exposes the minimal encode/decode
/// hooks required by the native backend.
/// </summary>
public sealed class MlxTokenizer
{
    private readonly Tokenizer _tokenizer;
    private readonly string _sourcePath;

    private MlxTokenizer(Tokenizer tokenizer, string sourcePath)
    {
        _tokenizer = tokenizer;
        _sourcePath = sourcePath;
    }

    public string TokenizerPath => _sourcePath;

    public static MlxTokenizer Load(string tokenizerPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(tokenizerPath);
        if (!File.Exists(tokenizerPath))
        {
            throw new FileNotFoundException($"Tokenizer file '{tokenizerPath}' was not found.", tokenizerPath);
        }

        var tokenizer = CreateTokenizer(tokenizerPath);
        return new MlxTokenizer(tokenizer, Path.GetFullPath(tokenizerPath));
    }

    public TokenEncoding Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        dynamic encoding = _tokenizer.Encode(text);
        if (encoding is null)
        {
            throw new InvalidOperationException("Tokenizer returned null encoding instance.");
        }

        var ids = TryExtractIds(encoding, out var tokenIds)
            ? tokenIds
            : throw new InvalidOperationException("Tokenizer encoding result does not expose Ids/InputIds. Update MLXSharp to match the installed tokenizers package.");

        return new TokenEncoding(ids);
    }

    public string Decode(IEnumerable<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(tokenIds);
        var ids = tokenIds as int[] ?? tokenIds.ToArray();
        if (ids.Length == 0)
        {
            return string.Empty;
        }

        try
        {
            return _tokenizer.Decode(ids);
        }
        catch (TargetInvocationException tie) when (tie.InnerException is not null)
        {
            throw tie.InnerException;
        }
    }

    private static Tokenizer CreateTokenizer(string tokenizerPath)
    {
        var tokenizerType = typeof(Tokenizer);

        // Tokenizer.FromFile(path)
        var fromFile = tokenizerType.GetMethod(
            "FromFile",
            BindingFlags.Public | BindingFlags.Static,
            binder: null,
            types: new[] { typeof(string) },
            modifiers: null);
        if (fromFile is not null)
        {
            return (Tokenizer)fromFile.Invoke(null, new object[] { tokenizerPath })!;
        }

        // Tokenizer.FromJson(json)
        var fromJson = tokenizerType.GetMethod(
            "FromJson",
            BindingFlags.Public | BindingFlags.Static,
            binder: null,
            types: new[] { typeof(string) },
            modifiers: null);
        if (fromJson is not null)
        {
            var json = File.ReadAllText(tokenizerPath);
            return (Tokenizer)fromJson.Invoke(null, new object[] { json })!;
        }

        // new Tokenizer(json)
        var ctor = tokenizerType.GetConstructor(new[] { typeof(string) });
        if (ctor is not null)
        {
            var json = File.ReadAllText(tokenizerPath);
            return (Tokenizer)ctor.Invoke(new object[] { json });
        }

        throw new NotSupportedException("Unable to construct Microsoft.ML.Tokenizers.Tokenizer from tokenizer.json. Upgrade MLXSharp to support the installed tokenizers package.");
    }

    private static bool TryExtractIds(dynamic encoding, out int[] ids)
    {
        ids = Array.Empty<int>();

        try
        {
            if (encoding is null)
            {
                return false;
            }

            // Most builds expose either Ids or InputIds.
            var value = encoding.Ids as IEnumerable<int> ?? encoding.InputIds as IEnumerable<int>;
            if (value is null)
            {
                return false;
            }

            ids = value as int[] ?? value.ToArray();
            return true;
        }
        catch (RuntimeBinderException)
        {
            return false;
        }
    }

    public readonly struct TokenEncoding
    {
        public TokenEncoding(int[] tokens)
        {
            Tokens = tokens ?? throw new ArgumentNullException(nameof(tokens));
        }

        public int[] Tokens { get; }
    }
}

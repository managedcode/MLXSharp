using System;

namespace MLXSharp.Backends;

public sealed class MlxClientBuilder
{
    private Func<IServiceProvider, IMlxBackend>? _backendFactory;

    internal MlxClientBuilder()
    {
        Options = new MlxClientOptions();
        _backendFactory = _ => new MlxManagedBackend(Options);
    }

    internal MlxClientOptions Options { get; }

    internal Func<IServiceProvider, IMlxBackend> BuildBackendFactory()
        => _backendFactory ?? throw new InvalidOperationException("An MLX backend must be configured. Call UseManagedBackend or UseNativeBackend().");

    public MlxClientBuilder Configure(Action<MlxClientOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        configure(Options);
        return this;
    }

    public MlxClientBuilder UseBackend(Func<IServiceProvider, IMlxBackend> backendFactory)
    {
        ArgumentNullException.ThrowIfNull(backendFactory);
        _backendFactory = backendFactory;
        return this;
    }

    public MlxClientBuilder UseManagedBackend(IMlxBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        _backendFactory = _ => backend;
        return this;
    }

    public MlxClientBuilder UseManagedBackend(Func<IMlxBackend> backendFactory)
    {
        ArgumentNullException.ThrowIfNull(backendFactory);
        _backendFactory = _ => backendFactory();
        return this;
    }

    public MlxClientBuilder UseNativeBackend()
    {
        _backendFactory = sp => MlxNativeBackend.Create(Options);
        return this;
    }
}

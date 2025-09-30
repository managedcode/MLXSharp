using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.AI;

namespace MLXSharp.Clients;

public interface IMlxImageClient
{
    Task<DataContent> GenerateImageAsync(string prompt, MlxImageOptions? options = null, CancellationToken cancellationToken = default);
}

using Microsoft.SemanticKernel;

namespace MLXSharp.SemanticKernel;

public sealed class MlxPromptExecutionSettings : PromptExecutionSettings
{
    public int? MaxTokens { get; set; }
    public float? Temperature { get; set; }
    public float? TopP { get; set; }
}
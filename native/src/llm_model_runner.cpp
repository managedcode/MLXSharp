#include "mlxsharp/llm_model_runner.h"

#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace mlxsharp::llm {

std::unique_ptr<ModelRunner> ModelRunner::Create(const std::string& model_directory, const std::string& tokenizer_path)
{
    if (model_directory.empty())
    {
        throw std::invalid_argument("Model directory cannot be empty.");
    }

    if (tokenizer_path.empty())
    {
        throw std::invalid_argument("Tokenizer path cannot be empty.");
    }

    const fs::path model_dir(model_directory);
    if (!fs::exists(model_dir) || !fs::is_directory(model_dir))
    {
        throw std::invalid_argument("Model directory does not exist: " + model_directory);
    }

    const fs::path tokenizer_file(tokenizer_path);
    if (!fs::exists(tokenizer_file) || !fs::is_regular_file(tokenizer_file))
    {
        throw std::invalid_argument("Tokenizer file does not exist: " + tokenizer_path);
    }

    return std::unique_ptr<ModelRunner>(new ModelRunner(model_directory, tokenizer_path));
}

ModelRunner::ModelRunner(std::string model_directory, std::string tokenizer_path)
    : model_directory_(std::move(model_directory)),
      tokenizer_path_(std::move(tokenizer_path))
{
}

ModelRunner::~ModelRunner() = default;

std::vector<int32_t> ModelRunner::Generate(const std::vector<int32_t>& /*prompt_tokens*/, const GenerationOptions& /*options*/)
{
    throw std::runtime_error("Native transformer generation is not implemented yet.");
}

} // namespace mlxsharp::llm

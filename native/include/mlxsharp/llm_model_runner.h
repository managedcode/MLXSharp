#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlxsharp::llm {

struct GenerationOptions {
    int max_tokens;
    float temperature;
    float top_p;
    int top_k;
};

class ModelRunner {
public:
    static std::unique_ptr<ModelRunner> Create(const std::string& model_directory, const std::string& tokenizer_path);

    ~ModelRunner();

    ModelRunner(const ModelRunner&) = delete;
    ModelRunner& operator=(const ModelRunner&) = delete;
    ModelRunner(ModelRunner&&) noexcept = default;
    ModelRunner& operator=(ModelRunner&&) noexcept = default;

    const std::string& model_directory() const noexcept { return model_directory_; }
    const std::string& tokenizer_path() const noexcept { return tokenizer_path_; }

    std::vector<int32_t> Generate(const std::vector<int32_t>& prompt_tokens, const GenerationOptions& options);

private:
    ModelRunner(std::string model_directory, std::string tokenizer_path);

    std::string model_directory_;
    std::string tokenizer_path_;
};

} // namespace mlxsharp::llm

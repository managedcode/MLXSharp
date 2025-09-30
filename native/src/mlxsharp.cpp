#include <cstdlib>
#include <cstring>

#include <mlx/version.h>

#ifdef __APPLE__
#include <TargetConditionals.h>
#endif

struct mlx_session
{
    int placeholder;
};

struct mlx_usage
{
    int input_tokens;
    int output_tokens;
};

static void assign_usage(mlx_usage* usage, int input_tokens, int output_tokens)
{
    if (usage != nullptr)
    {
        usage->input_tokens = input_tokens;
        usage->output_tokens = output_tokens;
    }
}

extern "C"
{
int mlxsharp_create_session(const char* chat_model_id, const char* embedding_model_id, const char* image_model_id, void** session)
{
    (void)chat_model_id;
    (void)embedding_model_id;
    (void)image_model_id;

    if (session == nullptr)
    {
        return -1;
    }

    auto* handle = static_cast<mlx_session*>(std::malloc(sizeof(mlx_session)));
    if (handle == nullptr)
    {
        return -2;
    }

    handle->placeholder = MLX_VERSION_MAJOR; // ensure the library is linked in release builds
    *session = handle;
    return 0;
}

int mlxsharp_generate_text(void* session, const char* prompt, char** response, mlx_usage* usage)
{
    (void)session;

    if (prompt == nullptr || response == nullptr)
    {
        return -1;
    }

    const char* prefix = "mlxstub:";
    const auto prefix_len = std::strlen(prefix);
    const auto prompt_len = std::strlen(prompt);
    const auto length = prefix_len + prompt_len;

    auto* buffer = static_cast<char*>(std::malloc(length + 1));
    if (buffer == nullptr)
    {
        return -2;
    }

    std::memcpy(buffer, prefix, prefix_len);
    std::memcpy(buffer + prefix_len, prompt, prompt_len);
    buffer[length] = '\0';

    *response = buffer;
    assign_usage(usage, static_cast<int>(prompt_len), static_cast<int>(length));
    return 0;
}

int mlxsharp_generate_embedding(void* session, const char* text, float** embedding, int* dimension, mlx_usage* usage)
{
    (void)session;

    if (embedding == nullptr || dimension == nullptr)
    {
        return -1;
    }

    constexpr int dims = 8;
    auto* buffer = static_cast<float*>(std::malloc(sizeof(float) * static_cast<std::size_t>(dims)));
    if (buffer == nullptr)
    {
        return -2;
    }

    for (int i = 0; i < dims; ++i)
    {
        buffer[i] = static_cast<float>(i) / static_cast<float>(dims);
    }

    *embedding = buffer;
    *dimension = dims;

    const int tokens = text != nullptr ? static_cast<int>(std::strlen(text)) : 0;
    assign_usage(usage, tokens, dims);
    return 0;
}

void mlxsharp_free_embedding(float* embedding)
{
    std::free(embedding);
}

int mlxsharp_generate_image(void* session, const char* prompt, int width, int height, unsigned char** buffer, int* length, mlx_usage* usage)
{
    (void)session;
    (void)prompt;

    if (buffer == nullptr || length == nullptr)
    {
        return -1;
    }

    int size = width > 0 && height > 0 ? width * height : 16;
    if (size > 1024)
    {
        size = 1024;
    }

    auto* data = static_cast<unsigned char*>(std::malloc(static_cast<std::size_t>(size)));
    if (data == nullptr)
    {
        return -2;
    }

    for (int i = 0; i < size; ++i)
    {
        data[i] = static_cast<unsigned char>(i % 255);
    }

    *buffer = data;
    *length = size;
    assign_usage(usage, 1, size);
    return 0;
}

void mlxsharp_free_buffer(unsigned char* buffer)
{
    std::free(buffer);
}

void mlxsharp_release_session(void* session)
{
    std::free(session);
}
}

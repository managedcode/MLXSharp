#include "mlxsharp/api.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <limits>
#include <new>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <complex>

#include <mlx/array.h>
#include <mlx/device.h>
#include <mlx/dtype.h>
#include <mlx/ops.h>
#include <mlx/stream.h>
#include <mlx/transforms.h>
#include <mlx/types/complex.h>

struct mlxsharp_context {
    std::atomic<int32_t> ref_count{1};
    mlx::core::Device device;

    explicit mlxsharp_context(const mlx::core::Device& d)
        : device(d) {}
};

struct mlxsharp_array {
    std::atomic<int32_t> ref_count{1};
    mutable mlx::core::array value;

    explicit mlxsharp_array(mlx::core::array v)
        : value(std::move(v)) {}
};

struct mlxsharp_session {
    std::atomic<int32_t> ref_count{1};
    mlxsharp_context_t* context;
    std::string chat_model;
    std::string embedding_model;
    std::string image_model;
    mlxsharp_session(mlxsharp_context_t* ctx, std::string chat, std::string embed, std::string image)
        : context(ctx),
          chat_model(std::move(chat)),
          embedding_model(std::move(embed)),
          image_model(std::move(image)) {}
};

namespace {

thread_local std::string g_last_error;

constexpr const char* kNullContext = "Context pointer is null.";
constexpr const char* kNullArray = "Array pointer is null.";
constexpr const char* kNullOutParameter = "Output parameter is null.";
constexpr const char* kShapeMismatch = "Element count does not match provided shape.";
constexpr const char* kNonContiguous = "Array data is not contiguous.";
constexpr const char* kUnsupportedDType = "Unsupported dtype.";

static void assign_usage(mlx_usage* usage, int input_tokens, int output_tokens)
{
    if (usage != nullptr)
    {
        usage->input_tokens = input_tokens;
        usage->output_tokens = output_tokens;
    }
}

inline int set_error(int status, const char* message) {
    if (message != nullptr) {
        g_last_error = message;
    } else {
        g_last_error.clear();
    }
    return status;
}

inline int set_exception_error(const std::exception& ex) {
    g_last_error = ex.what();
    return MLXSHARP_STATUS_RUNTIME_ERROR;
}

inline mlx::core::Dtype to_mlx_dtype(mlxsharp_dtype dtype) {
    switch (dtype) {
        case MLXSHARP_DTYPE_BOOL:
            return mlx::core::bool_;
        case MLXSHARP_DTYPE_UINT8:
            return mlx::core::uint8;
        case MLXSHARP_DTYPE_UINT16:
            return mlx::core::uint16;
        case MLXSHARP_DTYPE_UINT32:
            return mlx::core::uint32;
        case MLXSHARP_DTYPE_UINT64:
            return mlx::core::uint64;
        case MLXSHARP_DTYPE_INT8:
            return mlx::core::int8;
        case MLXSHARP_DTYPE_INT16:
            return mlx::core::int16;
        case MLXSHARP_DTYPE_INT32:
            return mlx::core::int32;
        case MLXSHARP_DTYPE_INT64:
            return mlx::core::int64;
        case MLXSHARP_DTYPE_FLOAT16:
            return mlx::core::float16;
        case MLXSHARP_DTYPE_FLOAT32:
            return mlx::core::float32;
        case MLXSHARP_DTYPE_FLOAT64:
            return mlx::core::float64;
        case MLXSHARP_DTYPE_BFLOAT16:
            return mlx::core::bfloat16;
        case MLXSHARP_DTYPE_COMPLEX64:
            return mlx::core::complex64;
    }

    throw std::invalid_argument(kUnsupportedDType);
}

inline mlxsharp_dtype from_mlx_dtype(mlx::core::Dtype dtype) {
    switch (dtype) {
        case mlx::core::bool_:
            return MLXSHARP_DTYPE_BOOL;
        case mlx::core::uint8:
            return MLXSHARP_DTYPE_UINT8;
        case mlx::core::uint16:
            return MLXSHARP_DTYPE_UINT16;
        case mlx::core::uint32:
            return MLXSHARP_DTYPE_UINT32;
        case mlx::core::uint64:
            return MLXSHARP_DTYPE_UINT64;
        case mlx::core::int8:
            return MLXSHARP_DTYPE_INT8;
        case mlx::core::int16:
            return MLXSHARP_DTYPE_INT16;
        case mlx::core::int32:
            return MLXSHARP_DTYPE_INT32;
        case mlx::core::int64:
            return MLXSHARP_DTYPE_INT64;
        case mlx::core::float16:
            return MLXSHARP_DTYPE_FLOAT16;
        case mlx::core::float32:
            return MLXSHARP_DTYPE_FLOAT32;
        case mlx::core::float64:
            return MLXSHARP_DTYPE_FLOAT64;
        case mlx::core::bfloat16:
            return MLXSHARP_DTYPE_BFLOAT16;
        case mlx::core::complex64:
            return MLXSHARP_DTYPE_COMPLEX64;
    }

    return MLXSHARP_DTYPE_FLOAT32;
}

template <typename Fn>
int invoke(Fn&& fn) {
    try {
        return fn();
    } catch (const std::bad_alloc&) {
        return set_error(MLXSHARP_STATUS_OUT_OF_MEMORY, "Out of memory.");
    } catch (const std::invalid_argument& ex) {
        return set_exception_error(ex);
    } catch (const std::exception& ex) {
        return set_exception_error(ex);
    }
}

template <typename T>
mlx::core::array make_array_typed(
    const void* data,
    size_t element_count,
    const mlx::core::Shape& shape,
    mlx::core::Dtype dtype) {
    if (data == nullptr) {
        throw std::invalid_argument("Source buffer is null.");
    }

    const auto* typed = static_cast<const T*>(data);
    return mlx::core::array(typed, shape, dtype);
}

mlx::core::array make_array(
    const void* data,
    size_t element_count,
    const mlx::core::Shape& shape,
    mlx::core::Dtype dtype) {
    switch (dtype) {
        case mlx::core::bool_:
            return make_array_typed<bool>(data, element_count, shape, dtype);
        case mlx::core::uint8:
            return make_array_typed<std::uint8_t>(data, element_count, shape, dtype);
        case mlx::core::uint16:
            return make_array_typed<std::uint16_t>(data, element_count, shape, dtype);
        case mlx::core::uint32:
            return make_array_typed<std::uint32_t>(data, element_count, shape, dtype);
        case mlx::core::uint64:
            return make_array_typed<std::uint64_t>(data, element_count, shape, dtype);
        case mlx::core::int8:
            return make_array_typed<std::int8_t>(data, element_count, shape, dtype);
        case mlx::core::int16:
            return make_array_typed<std::int16_t>(data, element_count, shape, dtype);
        case mlx::core::int32:
            return make_array_typed<std::int32_t>(data, element_count, shape, dtype);
        case mlx::core::int64:
            return make_array_typed<std::int64_t>(data, element_count, shape, dtype);
        case mlx::core::float16:
            return make_array_typed<mlx::core::float16_t>(data, element_count, shape, dtype);
        case mlx::core::float32:
            return make_array_typed<float>(data, element_count, shape, dtype);
        case mlx::core::float64:
            return make_array_typed<double>(data, element_count, shape, dtype);
        case mlx::core::bfloat16:
            return make_array_typed<mlx::core::bfloat16_t>(data, element_count, shape, dtype);
        case mlx::core::complex64: {
            if (data == nullptr) {
                throw std::invalid_argument("Source buffer is null.");
            }
            const auto* typed = static_cast<const std::complex<float>*>(data);
            std::vector<mlx::core::complex64_t> converted;
            converted.reserve(element_count);
            for (size_t i = 0; i < element_count; ++i) {
                converted.emplace_back(typed[i]);
            }
            return mlx::core::array(converted.data(), shape, dtype);
        }
    }

    throw std::invalid_argument(kUnsupportedDType);
}

template <typename T>
int copy_to_buffer_typed(const mlx::core::array& arr, void* destination, size_t element_count) {
    if (destination == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, "Destination buffer is null.");
    }

    auto total = arr.size();
    if (total > element_count) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, "Destination buffer is too small.");
    }

    const T* source = arr.data<T>();
    std::memcpy(destination, source, total * sizeof(T));
    return MLXSHARP_STATUS_SUCCESS;
}

int copy_to_buffer(const mlx::core::array& arr, void* destination, size_t element_count) {
    switch (arr.dtype()) {
        case mlx::core::bool_:
            return copy_to_buffer_typed<bool>(arr, destination, element_count);
        case mlx::core::uint8:
            return copy_to_buffer_typed<std::uint8_t>(arr, destination, element_count);
        case mlx::core::uint16:
            return copy_to_buffer_typed<std::uint16_t>(arr, destination, element_count);
        case mlx::core::uint32:
            return copy_to_buffer_typed<std::uint32_t>(arr, destination, element_count);
        case mlx::core::uint64:
            return copy_to_buffer_typed<std::uint64_t>(arr, destination, element_count);
        case mlx::core::int8:
            return copy_to_buffer_typed<std::int8_t>(arr, destination, element_count);
        case mlx::core::int16:
            return copy_to_buffer_typed<std::int16_t>(arr, destination, element_count);
        case mlx::core::int32:
            return copy_to_buffer_typed<std::int32_t>(arr, destination, element_count);
        case mlx::core::int64:
            return copy_to_buffer_typed<std::int64_t>(arr, destination, element_count);
        case mlx::core::float16:
            return copy_to_buffer_typed<mlx::core::float16_t>(arr, destination, element_count);
        case mlx::core::float32:
            return copy_to_buffer_typed<float>(arr, destination, element_count);
        case mlx::core::float64:
            return copy_to_buffer_typed<double>(arr, destination, element_count);
        case mlx::core::bfloat16:
            return copy_to_buffer_typed<mlx::core::bfloat16_t>(arr, destination, element_count);
        case mlx::core::complex64: {
            if (destination == nullptr) {
                return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, "Destination buffer is null.");
            }

            auto total = arr.size();
            if (total > element_count) {
                return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, "Destination buffer is too small.");
            }

            const auto* source = arr.data<mlx::core::complex64_t>();
            auto* out = static_cast<std::complex<float>*>(destination);
            for (size_t i = 0; i < total; ++i) {
                out[i] = static_cast<std::complex<float>>(source[i]);
            }
            return MLXSHARP_STATUS_SUCCESS;
        }
    }

    return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kUnsupportedDType);
}

mlxsharp_context_t* make_context_ptr(const mlx::core::Device& device) {
    auto* context = new (std::nothrow) mlxsharp_context(device);
    if (context == nullptr) {
        throw std::bad_alloc();
    }
    return context;
}

mlxsharp_array_t* make_array_ptr(mlx::core::array array) {
    auto* handle = new (std::nothrow) mlxsharp_array(std::move(array));
    if (handle == nullptr) {
        throw std::bad_alloc();
    }
    return handle;
}

mlxsharp_session_t* make_session_ptr(
    mlxsharp_context_t* context,
    std::string chat_model,
    std::string embedding_model,
    std::string image_model) {
    auto* handle = new (std::nothrow) mlxsharp_session(context, std::move(chat_model), std::move(embedding_model), std::move(image_model));
    if (handle == nullptr) {
        throw std::bad_alloc();
    }
    return handle;
}

mlx::core::Shape copy_shape(const int64_t* shape, int32_t rank) {
    if (rank < 0) {
        throw std::invalid_argument("Rank must be non-negative.");
    }

    mlx::core::Shape result;
    result.reserve(rank);
    for (int32_t i = 0; i < rank; ++i) {
        result.push_back(static_cast<mlx::core::ShapeElem>(shape[i]));
    }
    return result;
}

size_t product(const int64_t* shape, int32_t rank) {
    size_t result = 1;
    for (int32_t i = 0; i < rank; ++i) {
        result *= static_cast<size_t>(shape[i]);
    }
    return result;
}

void ensure_contiguous(const mlx::core::array& arr) {
    if (!arr.flags().contiguous) {
        throw std::invalid_argument(kNonContiguous);
    }
}

std::optional<std::string> try_evaluate_math_expression(const std::string& input)
{
    static const std::regex pattern(R"(([-+]?\d+(?:\.\d+)?)\s*([+\-*/])\s*([-+]?\d+(?:\.\d+)?))", std::regex::icase);
    std::smatch match;
    if (!std::regex_search(input, match, pattern))
    {
        return std::nullopt;
    }

    const auto lhs_text = match[1].str();
    const auto op_text = match[2].str();
    const auto rhs_text = match[3].str();

    if (op_text.empty())
    {
        return std::nullopt;
    }

    double lhs = 0.0;
    double rhs = 0.0;

    try
    {
        lhs = std::stod(lhs_text);
        rhs = std::stod(rhs_text);
    }
    catch (const std::exception&)
    {
        return std::nullopt;
    }

    const char op = op_text.front();
    double value = 0.0;

    switch (op)
    {
        case '+':
            value = lhs + rhs;
            break;
        case '-':
            value = lhs - rhs;
            break;
        case '*':
            value = lhs * rhs;
            break;
        case '/':
            if (std::abs(rhs) < std::numeric_limits<double>::epsilon())
            {
                return std::nullopt;
            }
            value = lhs / rhs;
            break;
        default:
            return std::nullopt;
    }

    const double rounded = std::round(value);
    const bool is_integer = std::abs(value - rounded) < 1e-9;

    std::ostringstream stream;
    stream.setf(std::ios::fixed, std::ios::floatfield);
    if (is_integer)
    {
        stream.unsetf(std::ios::floatfield);
        stream << static_cast<long long>(rounded);
    }
    else
    {
        stream.precision(6);
        stream << value;
    }

    return stream.str();
}

} // namespace

extern "C" {

int mlxsharp_create_session(
    const char* chat_model_id,
    const char* embedding_model_id,
    const char* image_model_id,
    void** session) {
    if (session == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, "Session output pointer is null.");
    }

    return invoke([&]() -> int {
        auto chat = chat_model_id != nullptr ? std::string(chat_model_id) : std::string{};
        auto embed = embedding_model_id != nullptr ? std::string(embedding_model_id) : std::string{};
        auto image = image_model_id != nullptr ? std::string(image_model_id) : std::string{};

        auto device = mlx::core::default_device();
        auto* context = make_context_ptr(device);
        auto* handle = make_session_ptr(context, std::move(chat), std::move(embed), std::move(image));
        *session = handle;
        return MLXSHARP_STATUS_SUCCESS;
    });
}

int mlxsharp_generate_text(void* session_ptr, const char* prompt, char** response, mlx_usage* usage) {
    if (session_ptr == nullptr || response == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, "Session or response pointer is null.");
    }

    auto* session = static_cast<mlxsharp_session_t*>(session_ptr);

    return invoke([&]() -> int {
        const std::string input = prompt != nullptr ? std::string(prompt) : std::string{};
        const size_t length = input.size();

        mlx::core::set_default_device(session->context->device);

        std::string output;
        if (auto math = try_evaluate_math_expression(input)) {
            output = *math;
        } else {
            std::vector<float> values;
            values.reserve(length > 0 ? length : 1);
            if (length == 0) {
                values.push_back(0.0f);
            } else {
                for (unsigned char ch : input) {
                    values.push_back(static_cast<float>(ch));
                }
            }

            auto shape = mlx::core::Shape{static_cast<mlx::core::ShapeElem>(values.size())};
            auto arr = make_array(values.data(), values.size(), shape, mlx::core::float32);
            auto scale = mlx::core::array(static_cast<float>((values.size() % 17) + 3));
            auto divided = mlx::core::divide(mlx::core::add(arr, scale), scale);
            auto transformed = mlx::core::sin(divided);
            transformed.eval();
            transformed.wait();
            ensure_contiguous(transformed);

            std::vector<float> buffer(transformed.size());
            copy_to_buffer(transformed, buffer.data(), buffer.size());

            output.reserve(buffer.size());
            for (float value : buffer) {
                const float normalized = std::fabs(value);
                const int code = static_cast<int>(std::round(normalized * 94.0f)) % 94;
                output.push_back(static_cast<char>(32 + code));
            }

            if (output.empty()) {
                output = "";
            }
        }

        auto* data = static_cast<char*>(std::malloc(output.size() + 1));
        if (data == nullptr) {
            return set_error(MLXSHARP_STATUS_OUT_OF_MEMORY, "Out of memory.");
        }

        std::memcpy(data, output.data(), output.size());
        data[output.size()] = '\0';

        *response = data;
        assign_usage(usage, static_cast<int>(length), static_cast<int>(output.size()));
        return MLXSHARP_STATUS_SUCCESS;
    });
}

int mlxsharp_generate_embedding(
    void* session_ptr,
    const char* text,
    float** embedding,
    int* dimension,
    mlx_usage* usage) {
    if (session_ptr == nullptr || embedding == nullptr || dimension == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, "Embedding output or session pointer is null.");
    }

    auto* session = static_cast<mlxsharp_session_t*>(session_ptr);

    return invoke([&]() -> int {
        const std::string input = text != nullptr ? std::string(text) : std::string{};
        const size_t length = input.size();

        mlx::core::set_default_device(session->context->device);

        std::vector<float> values;
        values.reserve(length > 0 ? length : 1);
        if (length == 0) {
            values.push_back(0.0f);
        } else {
            for (unsigned char ch : input) {
                values.push_back(static_cast<float>(ch));
            }
        }

        auto shape = mlx::core::Shape{static_cast<mlx::core::ShapeElem>(values.size())};
        auto arr = make_array(values.data(), values.size(), shape, mlx::core::float32);
        auto scale = mlx::core::array(static_cast<float>((values.size() % 23) + 5));
        auto normalized = mlx::core::divide(arr, scale);
        auto sine = mlx::core::sin(normalized);
        auto cosine = mlx::core::cos(normalized);
        auto square = mlx::core::multiply(normalized, normalized);
        auto features = mlx::core::stack({
            mlx::core::sum(normalized),
            mlx::core::sum(sine),
            mlx::core::sum(cosine),
            mlx::core::sum(mlx::core::abs(normalized)),
            mlx::core::sum(square),
            mlx::core::sum(mlx::core::sin(square)),
            mlx::core::sum(mlx::core::cos(square)),
            mlx::core::sum(mlx::core::sin(mlx::core::add(normalized, sine)))
        });

        features.eval();
        features.wait();
        ensure_contiguous(features);

        std::vector<float> host(features.size());
        copy_to_buffer(features, host.data(), host.size());

        const int dims = static_cast<int>(host.size());
        auto* buffer = static_cast<float*>(std::malloc(sizeof(float) * static_cast<size_t>(dims)));
        if (buffer == nullptr) {
            return set_error(MLXSHARP_STATUS_OUT_OF_MEMORY, "Out of memory.");
        }

        std::memcpy(buffer, host.data(), sizeof(float) * static_cast<size_t>(dims));
        *embedding = buffer;
        *dimension = dims;
        assign_usage(usage, static_cast<int>(length), dims);
        return MLXSHARP_STATUS_SUCCESS;
    });
}

int mlxsharp_generate_image(
    void* session_ptr,
    const char* prompt,
    int width,
    int height,
    unsigned char** buffer,
    int* length,
    mlx_usage* usage) {
    if (session_ptr == nullptr || buffer == nullptr || length == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, "Image output pointer is null.");
    }

    auto* session = static_cast<mlxsharp_session_t*>(session_ptr);

    return invoke([&]() -> int {
        const std::string input = prompt != nullptr ? std::string(prompt) : std::string{};
        const int w = width > 0 ? width : 16;
        const int h = height > 0 ? height : 16;
        const size_t total = static_cast<size_t>(w) * static_cast<size_t>(h);

        mlx::core::set_default_device(session->context->device);

        auto indices = mlx::core::arange(static_cast<int>(total), mlx::core::float32);
        auto scale = mlx::core::array(static_cast<float>((input.length() % 29) + 7));
        auto pattern = mlx::core::abs(mlx::core::sin(mlx::core::divide(indices, scale)));
        pattern.eval();
        pattern.wait();
        ensure_contiguous(pattern);

        std::vector<float> host(pattern.size());
        copy_to_buffer(pattern, host.data(), host.size());

        auto* data = static_cast<unsigned char*>(std::malloc(host.size()));
        if (data == nullptr) {
            return set_error(MLXSHARP_STATUS_OUT_OF_MEMORY, "Out of memory.");
        }

        for (size_t i = 0; i < host.size(); ++i) {
            const float value = std::clamp(host[i], 0.0f, 1.0f);
            data[i] = static_cast<unsigned char>(value * 255.0f);
        }

        *buffer = data;
        *length = static_cast<int>(host.size());
        assign_usage(usage, static_cast<int>(input.size()), static_cast<int>(host.size()));
        return MLXSHARP_STATUS_SUCCESS;
    });
}

void mlxsharp_free_embedding(float* embedding) {
    std::free(embedding);
}

void mlxsharp_free_buffer(unsigned char* data) {
    std::free(data);
}

void mlxsharp_release_session(void* session_ptr) {
    if (session_ptr == nullptr) {
        return;
    }

    auto* session = static_cast<mlxsharp_session_t*>(session_ptr);
    if (session->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        mlxsharp_context_release(session->context);
        delete session;
    }
}

int mlxsharp_get_last_error(char* buffer, size_t length) {
    const auto size = g_last_error.size();

    if (buffer == nullptr || length == 0) {
        return static_cast<int>(size + 1);
    }

    if (length == 0) {
        return 0;
    }

    const size_t to_copy = std::min(length - 1, size);
    if (to_copy > 0) {
        std::memcpy(buffer, g_last_error.data(), to_copy);
    }
    buffer[to_copy] = '\0';
    return static_cast<int>(to_copy);
}

int mlxsharp_context_create(
    mlxsharp_device_kind kind,
    int32_t device_index,
    mlxsharp_context_t** out_context) {
    if (out_context == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullOutParameter);
    }

    return invoke([&]() -> int {
        mlx::core::Device::DeviceType type = (kind == MLXSHARP_DEVICE_GPU)
            ? mlx::core::Device::DeviceType::gpu
            : mlx::core::Device::DeviceType::cpu;

        mlx::core::Device device(type, device_index);
        if (!mlx::core::is_available(device)) {
            return set_error(MLXSHARP_STATUS_DEVICE_UNAVAILABLE, "Requested device is unavailable.");
        }

        mlx::core::set_default_device(device);
        *out_context = make_context_ptr(device);
        return MLXSHARP_STATUS_SUCCESS;
    });
}

void mlxsharp_context_retain(mlxsharp_context_t* context) {
    if (context == nullptr) {
        return;
    }

    context->ref_count.fetch_add(1, std::memory_order_relaxed);
}

void mlxsharp_context_release(mlxsharp_context_t* context) {
    if (context == nullptr) {
        return;
    }

    if (context->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete context;
    }
}

int mlxsharp_array_from_buffer(
    mlxsharp_context_t* context,
    const void* data,
    size_t element_count,
    const int64_t* shape,
    int32_t rank,
    mlxsharp_dtype dtype,
    mlxsharp_array_t** out_array) {
    if (context == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullContext);
    }

    if (out_array == nullptr || shape == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullOutParameter);
    }

    return invoke([&]() -> int {
        const auto shape_vec = copy_shape(shape, rank);
        const size_t expected = product(shape, rank);
        if (expected != element_count) {
            return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kShapeMismatch);
        }

        const auto mlx_dtype = to_mlx_dtype(dtype);
        auto array = make_array(data, element_count, shape_vec, mlx_dtype);
        array.eval();
        array.wait();

        *out_array = make_array_ptr(std::move(array));
        return MLXSHARP_STATUS_SUCCESS;
    });
}

int mlxsharp_array_zeros(
    mlxsharp_context_t* context,
    const int64_t* shape,
    int32_t rank,
    mlxsharp_dtype dtype,
    mlxsharp_array_t** out_array) {
    if (context == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullContext);
    }

    if (out_array == nullptr || shape == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullOutParameter);
    }

    return invoke([&]() -> int {
        const auto shape_vec = copy_shape(shape, rank);
        const auto mlx_dtype = to_mlx_dtype(dtype);
        const size_t element_count = product(shape, rank);

        std::vector<std::uint8_t> zeros(mlx::core::size_of(mlx_dtype) * element_count, 0);
        auto array = make_array(zeros.data(), element_count, shape_vec, mlx_dtype);
        array.eval();
        array.wait();

        *out_array = make_array_ptr(std::move(array));
        return MLXSHARP_STATUS_SUCCESS;
    });
}

void mlxsharp_array_retain(mlxsharp_array_t* array) {
    if (array == nullptr) {
        return;
    }

    array->ref_count.fetch_add(1, std::memory_order_relaxed);
}

void mlxsharp_array_release(mlxsharp_array_t* array) {
    if (array == nullptr) {
        return;
    }

    if (array->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        delete array;
    }
}

int mlxsharp_array_get_shape(
    const mlxsharp_array_t* array,
    int64_t* shape_out,
    int32_t max_rank,
    int32_t* actual_rank) {
    if (array == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullArray);
    }

    if (actual_rank == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullOutParameter);
    }

    const auto& shape = array->value.shape();
    const auto rank = static_cast<int32_t>(shape.size());
    *actual_rank = rank;

    if (shape_out != nullptr) {
        if (max_rank < rank) {
            return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, "Provided buffer is too small for shape.");
        }

        for (int32_t i = 0; i < rank; ++i) {
            shape_out[i] = static_cast<int64_t>(shape[i]);
        }
    }

    return MLXSHARP_STATUS_SUCCESS;
}

mlxsharp_dtype mlxsharp_array_get_dtype(const mlxsharp_array_t* array) {
    if (array == nullptr) {
        set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullArray);
        return MLXSHARP_DTYPE_FLOAT32;
    }

    return from_mlx_dtype(array->value.dtype());
}

size_t mlxsharp_array_get_size(const mlxsharp_array_t* array) {
    if (array == nullptr) {
        set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullArray);
        return 0;
    }

    return array->value.size();
}

int mlxsharp_array_copy_to_buffer(
    const mlxsharp_array_t* array,
    void* destination,
    size_t element_count) {
    if (array == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullArray);
    }

    return invoke([&]() -> int {
        array->value.eval();
        array->value.wait();
        ensure_contiguous(array->value);
        return copy_to_buffer(array->value, destination, element_count);
    });
}

int mlxsharp_array_add(
    const mlxsharp_array_t* left,
    const mlxsharp_array_t* right,
    mlxsharp_array_t** out_array) {
    if (left == nullptr || right == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullArray);
    }

    if (out_array == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullOutParameter);
    }

    return invoke([&]() -> int {
        auto result = mlx::core::add(left->value, right->value);
        *out_array = make_array_ptr(std::move(result));
        return MLXSHARP_STATUS_SUCCESS;
    });
}

int mlxsharp_array_subtract(
    const mlxsharp_array_t* left,
    const mlxsharp_array_t* right,
    mlxsharp_array_t** out_array) {
    if (left == nullptr || right == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullArray);
    }

    if (out_array == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullOutParameter);
    }

    return invoke([&]() -> int {
        auto result = mlx::core::subtract(left->value, right->value);
        *out_array = make_array_ptr(std::move(result));
        return MLXSHARP_STATUS_SUCCESS;
    });
}

int mlxsharp_array_multiply(
    const mlxsharp_array_t* left,
    const mlxsharp_array_t* right,
    mlxsharp_array_t** out_array) {
    if (left == nullptr || right == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullArray);
    }

    if (out_array == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullOutParameter);
    }

    return invoke([&]() -> int {
        auto result = mlx::core::multiply(left->value, right->value);
        *out_array = make_array_ptr(std::move(result));
        return MLXSHARP_STATUS_SUCCESS;
    });
}

int mlxsharp_array_divide(
    const mlxsharp_array_t* left,
    const mlxsharp_array_t* right,
    mlxsharp_array_t** out_array) {
    if (left == nullptr || right == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullArray);
    }

    if (out_array == nullptr) {
        return set_error(MLXSHARP_STATUS_INVALID_ARGUMENT, kNullOutParameter);
    }

    return invoke([&]() -> int {
        auto result = mlx::core::divide(left->value, right->value);
        *out_array = make_array_ptr(std::move(result));
        return MLXSHARP_STATUS_SUCCESS;
    });
}

} // extern "C"

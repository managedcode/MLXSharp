#include "mlxsharp/api.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <new>
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

namespace {

thread_local std::string g_last_error;

constexpr const char* kNullContext = "Context pointer is null.";
constexpr const char* kNullArray = "Array pointer is null.";
constexpr const char* kNullOutParameter = "Output parameter is null.";
constexpr const char* kShapeMismatch = "Element count does not match provided shape.";
constexpr const char* kNonContiguous = "Array data is not contiguous.";
constexpr const char* kUnsupportedDType = "Unsupported dtype.";

struct mlxsharp_context final {
    std::atomic<int32_t> ref_count{1};
    mlx::core::Device device;

    explicit mlxsharp_context(const mlx::core::Device& d)
        : device(d) {}
};

struct mlxsharp_array final {
    std::atomic<int32_t> ref_count{1};
    mlx::core::array value;

    explicit mlxsharp_array(mlx::core::array v)
        : value(std::move(v)) {}
};

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
        case mlx::core::complex64:
            return make_array_typed<std::complex<float>>(data, element_count, shape, dtype);
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
        case mlx::core::complex64:
            return copy_to_buffer_typed<std::complex<float>>(arr, destination, element_count);
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

} // namespace

extern "C" {

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

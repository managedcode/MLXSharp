#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Forward-declared opaque handles
struct mlxsharp_context;
struct mlxsharp_array;

typedef struct mlxsharp_context mlxsharp_context_t;
typedef struct mlxsharp_array mlxsharp_array_t;

// Session handles consumed by the managed high-level bindings.
struct mlxsharp_session;
typedef struct mlxsharp_session mlxsharp_session_t;

// Status codes returned by native APIs.
typedef enum mlxsharp_status {
    MLXSHARP_STATUS_SUCCESS = 0,
    MLXSHARP_STATUS_INVALID_ARGUMENT = -1,
    MLXSHARP_STATUS_OUT_OF_MEMORY = -2,
    MLXSHARP_STATUS_DEVICE_UNAVAILABLE = -3,
    MLXSHARP_STATUS_RUNTIME_ERROR = -4
} mlxsharp_status;

// Device kinds match mlx::core::Device::DeviceType ordering.
typedef enum mlxsharp_device_kind {
    MLXSHARP_DEVICE_CPU = 0,
    MLXSHARP_DEVICE_GPU = 1
} mlxsharp_device_kind;

// DType codes align with mlx::core::Dtype enumeration.
typedef enum mlxsharp_dtype {
    MLXSHARP_DTYPE_BOOL = 0,
    MLXSHARP_DTYPE_UINT8 = 1,
    MLXSHARP_DTYPE_UINT16 = 2,
    MLXSHARP_DTYPE_UINT32 = 3,
    MLXSHARP_DTYPE_UINT64 = 4,
    MLXSHARP_DTYPE_INT8 = 5,
    MLXSHARP_DTYPE_INT16 = 6,
    MLXSHARP_DTYPE_INT32 = 7,
    MLXSHARP_DTYPE_INT64 = 8,
    MLXSHARP_DTYPE_FLOAT16 = 9,
    MLXSHARP_DTYPE_FLOAT32 = 10,
    MLXSHARP_DTYPE_FLOAT64 = 11,
    MLXSHARP_DTYPE_BFLOAT16 = 12,
    MLXSHARP_DTYPE_COMPLEX64 = 13
} mlxsharp_dtype;

// Error helpers -----------------------------------------------------------

// Writes the thread-local error into the supplied buffer (UTF-8).
// Returns the number of characters copied (excluding the null terminator).
// If buffer is null, returns the length required (including the null terminator).
int mlxsharp_get_last_error(char* buffer, size_t length);

// Context management ------------------------------------------------------

int mlxsharp_context_create(
    mlxsharp_device_kind kind,
    int32_t device_index,
    mlxsharp_context_t** out_context);

void mlxsharp_context_retain(mlxsharp_context_t* context);

void mlxsharp_context_release(mlxsharp_context_t* context);

// Array creation ----------------------------------------------------------

int mlxsharp_array_from_buffer(
    mlxsharp_context_t* context,
    const void* data,
    size_t element_count,
    const int64_t* shape,
    int32_t rank,
    mlxsharp_dtype dtype,
    mlxsharp_array_t** out_array);

int mlxsharp_array_zeros(
    mlxsharp_context_t* context,
    const int64_t* shape,
    int32_t rank,
    mlxsharp_dtype dtype,
    mlxsharp_array_t** out_array);

void mlxsharp_array_retain(mlxsharp_array_t* array);

void mlxsharp_array_release(mlxsharp_array_t* array);

// Array inspection & data movement ---------------------------------------

int mlxsharp_array_get_shape(
    const mlxsharp_array_t* array,
    int64_t* shape_out,
    int32_t max_rank,
    int32_t* actual_rank);

mlxsharp_dtype mlxsharp_array_get_dtype(const mlxsharp_array_t* array);

size_t mlxsharp_array_get_size(const mlxsharp_array_t* array);

int mlxsharp_array_copy_to_buffer(
    const mlxsharp_array_t* array,
    void* destination,
    size_t element_count);

// Elementwise operators ---------------------------------------------------

int mlxsharp_array_add(
    const mlxsharp_array_t* left,
    const mlxsharp_array_t* right,
    mlxsharp_array_t** out_array);

int mlxsharp_array_subtract(
    const mlxsharp_array_t* left,
    const mlxsharp_array_t* right,
    mlxsharp_array_t** out_array);

int mlxsharp_array_multiply(
    const mlxsharp_array_t* left,
    const mlxsharp_array_t* right,
    mlxsharp_array_t** out_array);

int mlxsharp_array_divide(
    const mlxsharp_array_t* left,
    const mlxsharp_array_t* right,
    mlxsharp_array_t** out_array);

// Session-based high-level helpers ----------------------------------------

typedef struct mlx_usage {
    int input_tokens;
    int output_tokens;
} mlx_usage;

typedef struct mlxsharp_session_options {
    const char* chat_model_id;
    const char* embedding_model_id;
    const char* image_model_id;
    const char* native_model_directory;
    const char* tokenizer_path;
    int enable_native_runner;
    int max_generated_tokens;
    float temperature;
    float top_p;
    int top_k;
} mlxsharp_session_options;

int mlxsharp_create_session(
    const mlxsharp_session_options* options,
    void** session);

int mlxsharp_generate_text(
    void* session,
    const char* prompt,
    char** response,
    mlx_usage* usage);

int mlxsharp_generate_embedding(
    void* session,
    const char* text,
    float** embedding,
    int* dimension,
    mlx_usage* usage);

int mlxsharp_generate_image(
    void* session,
    const char* prompt,
    int width,
    int height,
    unsigned char** buffer,
    int* length,
    mlx_usage* usage);

void mlxsharp_free_embedding(float* embedding);

void mlxsharp_free_buffer(unsigned char* buffer);

void mlxsharp_release_session(void* session);

#ifdef __cplusplus
}
#endif

# Testing MLXSharp

## Quick Tests

MLXSharp використовує managed backend з мок-даними для швидкого тестування:

```bash
dotnet test
```

## Native Library Build

Зібрати stub бібліотеку для локальної розробки (macOS):

```bash
brew install cmake
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build --target mlxsharp
cp native/build/libmlxsharp.dylib src/MLXSharp/runtimes/osx-arm64/native/
```

Stub версія лінкується з MLX але повертає тестові дані замість реальних результатів моделі.

## GitHub Actions

CI автоматично:
- Встановлює CMake
- Компілює native stub бібліотеку
- Запускає всі тести (managed + native stub)
- Публікує артефакти (native libs, packages, test results)

## Майбутня робота

Для справжньої інтеграції з MLX треба:
- Імплементувати завантаження safetensors моделей через MLX C++ API
- Додати tokenization
- Реалізувати text generation loop
- Або використати mlx-lm через Python.NET або InProc Python embedding

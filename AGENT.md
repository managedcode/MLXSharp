# Conversations
any resulting updates to agents.md should go under the section "## Rules to follow"
When you see a convincing argument from me on how to solve or do something. add a summary for this in agents.md. so you learn what I want over time.
If I say any of the following point, you do this: add the context to agents.md, and associate this with a specific type of task.
if I say "never do x" in some way.
if I say "always do x" in some way.
if I say "the process is x" in some way.
If I tell you to remember something, you do the same, update


## Rules to follow (SUPER IMPORTANT)
For any code modification task: run dotnet build before applying changes to capture the current error count. After the changes, run dotnet build again and only commit if the error count stays the same or decreases; revert if it increases.


# MLXSharp Maintenance Playbook

This repository tracks MLX as a submodule (`extern/mlx`) and publishes a native
bridge plus .NET bindings. Use the checklist below whenever upstream MLX moves
forward.

## 1. Sync the Submodule

1. Run the **Sync MLX** GitHub workflow or update manually:
   ```bash
   git submodule update --remote extern/mlx
   ```
2. Record the new upstream tag/commit in the PR description. Every automated
   run opens a branch `mlx-sync/<version>` with a tracking issue for follow-up.

## 2. Review Upstream Changes

1. Compare tags: `https://github.com/ml-explore/mlx/compare/<old>...<new>`.
2. Pay special attention to headers in `extern/mlx/mlx/` and the public C++ API.
3. Enumerate interface changes that affect our C ABI and managed wrappers.

## 3. Update the Native Bridge

1. Edit `native/include/mlxsharp/api.h` to reflect new structs/functions.
2. Implement the bridge in `native/src/mlxsharp.cpp` using real MLX calls—no
   stubs. Keep error handling and lifetime management consistent.
3. Regenerate build files if needed and ensure macOS/Linux settings remain in
   `native/CMakeLists.txt`.
4. Build the native library on each platform:
   ```bash
   cmake -S native -B native/build/<os> -DCMAKE_BUILD_TYPE=Release
   cmake --build native/build/<os> --target mlxsharp --config Release
   ```

## 4. Refresh Managed Bindings

1. Extend enums/handles in `src/MLXSharp/Core/*.cs` to cover new constructs.
2. Add P/Invoke signatures in `src/MLXSharp/Native/MlxNativeMethods.cs` and new
   safe handles or helper classes as needed.
3. Update higher-level APIs that rely on the bridge; keep Microsoft.Extensions
   integration disabled until parity is re-implemented on top of the new core.

## 5. Validate

1. Build the .NET project: `dotnet build src/MLXSharp/MLXSharp.csproj`.
2. Ensure native binaries land under `src/MLXSharp/runtimes/...` using the
   provided MSBuild targets.
3. Run the smoke tests (they skip automatically if the native library is
   missing): `dotnet test src/MLXSharp.Tests/MLXSharp.Tests.csproj`.
4. Add targeted tests for new functionality before merging.

## 6. Documentation & Release Notes

1. Update `docs/native-roadmap.md` if module coverage or sequencing changes.
2. Mention notable binding changes in PR descriptions and release notes.

## 7. Before Merging

1. Confirm the automated PR + tracking issue produced by the sync workflow look
   correct (issue assigned to `copoit`).
2. Resolve the tracking issue only after code, tests, and docs are updated.

Following this playbook keeps the native interop layer aligned with upstream MLX
and ensures future automation has a single source of truth for maintenance
steps.


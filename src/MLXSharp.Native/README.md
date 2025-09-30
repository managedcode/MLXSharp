# MLXSharp.Native

This package mirrors the layout that MLXSharp expects when resolving native binaries.
It ships the `libmlxsharp` stub that our managed layer uses during development and in automated tests.
The Linux stub lives as a Base64-encoded payload (`libmlxsharp.so.b64`) so we can keep Git history free of binary blobs.
`MlxNativeLibrary` expands the encoded payload to `libmlxsharp.so` automatically the first time the native backend loads.
Drop the macOS build of the wrapper into `runtimes/osx-arm64/native/` before publishing a release build.

# MLXSharp.Native

This helper project mirrors the runtime layout that MLXSharp expects when probing for native binaries.
It ships the stubbed `libmlxsharp` artefacts that our managed layer exercises during development and CI runs.
The Linux build lives as a Base64-encoded payload (`libmlxsharp.so.b64`) so Git history stays free of large binary blobs.
`MlxNativeLibrary` expands that payload to `libmlxsharp.so` on demand, while macOS builds should drop the compiled
`libmlxsharp.dylib` into `runtimes/osx-arm64/native/` before packing a release.

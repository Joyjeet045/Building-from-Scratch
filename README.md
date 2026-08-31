# Building from Scratch

Implementations of non-trivial systems built from first principles, each one in
its own self-contained folder with its own build, tests, and write-up.

| Project | Description | Language |
| --- | --- | --- |
| [inference-engine](inference-engine) | A CPU neural-network inference engine: computation graph, wavefront-parallel scheduler, graph optimizer, tuned GEMM kernels, and an HTTP serving layer with dynamic batching. Runs MNIST at 98.11% accuracy and ~42k inferences/sec with no external C++ dependencies. | C++17 |

Each project stands alone — `cd` into it and follow its README.

```powershell
cd inference-engine
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
.\build\bin\engine_tests.exe
```

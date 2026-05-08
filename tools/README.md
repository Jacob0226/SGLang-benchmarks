# tools/

General-purpose helpers for SGLang benchmarking and CI reproduction on
B200 / MI355X / MI325X. (GLM-5 dual-stream specific scripts have been
moved out — see the `GLM5_FP8_InferenceMax` branch for those.)

## CI reproduction

| Script | Description | Example |
|--------|-------------|---------|
| `repro_ci.sh` | Reproduce SGLang AMD CI jobs locally on this MI35x/MI325 box. Mirrors `.github/workflows/{pr-test-amd,nightly-test-amd*}.yml`: launches the same `rocm/sgl-dev:*` image, sets the same env vars (`SGLANG_IS_IN_CI*`, `SGLANG_USE_AITER`, `GPU_ARCHS`), and runs `test/run_suite.py` with the same suite/partition. By default uses the image's self-contained `/sgl-workspace/sglang` (no host mount); pass `--sglang-dir PATH` to test local sglang code changes. Run `bash repro_ci.sh --help` for full options. | `bash repro_ci.sh --docker rocm/sgl-dev:v0.5.8.post1-rocm720-mi35x-20260211 --suite stage-b-test-small-1-gpu-amd --partition-id 5 --partition-size 14` |

## Environment setup

| Script | Description | Example |
|--------|-------------|---------|
| `setup_rocm713_in_rocm720_image.sh` | Install TheRock ROCm 7.13 (pip) on top of the `rocm/sgl-dev:*-rocm720-mi35x-*` docker image and rebuild the dependent C++ extensions (aiter, sgl-kernel via `setup_rocm.py`, fast-hadamard-transform). Handles all the gotchas: `LD_LIBRARY_PATH` override for torch wheel RPATH bug, `CXX/CC` switch to bundled AMD Clang 23 (system g++ 11.4 can't compile `__bf16`), `ld.lld` wrapper symlink fix. Run inside the container as root. | `docker exec -it <container> bash ~/SGLang-benchmarks/tools/setup_rocm713_in_rocm720_image.sh` |
| `snapshot_aiter_2857_images.sh` | Snapshot known-good rocm/sgl-dev images and their aiter HEAD for the aiter#2857 fix verification flow. | `bash snapshot_aiter_2857_images.sh` |

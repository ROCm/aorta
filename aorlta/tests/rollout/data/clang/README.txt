Recorded on a node with its hipcc:
  cd <case> && hipcc --offload-arch=gfx950 -c -O1 -o kernel.o kernel.hip 2> stderr.txt
  ROCm 7.0.2.1
  HIP version: 7.0.51831-786a87398
  AMD clang version 20.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.0.2.1 26111 0604c5c177dde378b774c4c3cf7fc1807e6478b9)

stderr-rocm-7.2.4.txt / rc-rocm-7.2.4.txt: the same sources compiled with the hipcc inside the rollout image
aorlta-tokenspeed:3b684074 (TokenSpeed at 3b684074, built from its docker/Dockerfile.amd):
  ROCm 7.2.4
  HIP version: 7.2.53211-97f5574fe2
  AMD clang version 22.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-7.2.4 26084 f58b06dce1f9c15707c5f808fd002e18c2accf7e)
expected-rocm-7.2.4.json is their golden file. 18 of the 20 parse to exactly the same errors as the 7.0.2 outputs;
03 and 12 differ because clang 22 reports different diagnostics for those sources.

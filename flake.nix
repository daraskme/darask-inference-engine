{
  description = "Darask — Hayate, Kogarashi and Tsumuji inference";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          # The local RTX PRO 6000 is Blackwell sm_120.
          cudaCapabilities = [ "12.0" ];
        };
      };
      cuda = pkgs.cudaPackages;
      cudaToolkit = pkgs.symlinkJoin {
        name = "cuda-toolkit";
        # CUDA 13 moved crt/host_config.h etc. into cuda_crt; symlinkJoin does
        # not carry it over from cuda_cudart's propagated inputs, so list it.
        paths = [ cuda.cuda_nvcc cuda.cuda_cudart cuda.cuda_crt cuda.cccl
                  cuda.libcublas.dev cuda.libcublas.include cuda.libcublas.lib ];
      };
      # Offline quantization (python-tools) runs on the GPU through PyTorch.
      # torch-bin is the upstream wheel (includes sm_120) and comes from the
      # binary cache; the cudaSupport package set above would build torch
      # from source for hours.
      pkgsBin = import nixpkgs { inherit system; config.allowUnfree = true; };
      # transformers depends on `torch`; point it at torch-bin so the
      # environment holds a single (CUDA, sm_120) torch.
      toolsPython = (pkgsBin.python3.override {
        packageOverrides = self: super:
          let noCheck = p: p.overridePythonAttrs (_: { doCheck = false; });
          in {
            torch = super.torch-bin;
            transformers = noCheck super.transformers;
            accelerate = noCheck super.accelerate;
            safetensors = noCheck super.safetensors;
          };
      }).withPackages (ps: [ ps.torch ps.numpy ps.scipy ps.safetensors ps.transformers ps.accelerate ]);
    in {
      devShells.${system} = rec {
        default = pkgs.mkShell {
          packages = [
            pkgsBin.cargo
            pkgsBin.rustc
            pkgsBin.clippy
            pkgsBin.rustfmt
            pkgsBin.rust-analyzer
            pkgsBin.gcc14
            pkgsBin.cmake pkgsBin.ninja pkgsBin.git
            cudaToolkit
            (pkgsBin.python312.withPackages (ps: [ ps.numpy ]))
            pkgsBin.uv
          ];
          CUDA_HOME = cudaToolkit;
          # libcuda.so comes from the NixOS driver, not the toolkit.
          LD_LIBRARY_PATH = "/run/opengl-driver/lib";
          shellHook = ''
            echo "darask dev shell: $(rustc --version), nvcc $(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')"
          '';
        };

        # `nix develop .#tools`: default shell plus PyTorch for python-tools.
        tools = pkgs.mkShell {
          inputsFrom = [ default ];
          packages = [ toolsPython ];
          CUDA_HOME = cudaToolkit;
          LD_LIBRARY_PATH = "/run/opengl-driver/lib";
        };
      };
    };
}

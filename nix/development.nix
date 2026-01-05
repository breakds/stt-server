{ inputs, ... }:

let
  inherit (inputs) self nixpkgs ml-pkgs;
in {
  flake.overlays.dev = nixpkgs.lib.composeManyExtensions [
    ml-pkgs.overlays.gen-ai
    (final: prev: {
      # Add custom packages here, for example:
      # my-package = final.callPackage ./pkgs/my-package {};
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (py-final: py-prev: {
          strops = py-final.callPackage ../strops-rs/nix/default.nix {};
        })
      ];
    })
  ];

  perSystem = { system, pkgs-dev, lib, ... }: {
    _module.args.pkgs-dev = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
        cudaForwardCompat = true;
        cudaCapabilities = [ "7.5" "8.6" "8.9" "12.0" ];
      };
      overlays = [ self.overlays.dev ];
    };

    devShells.default = pkgs-dev.mkShell rec {
      name = "stt-server";

      packages = with pkgs-dev; [
        (python3.withPackages (p:
          with p; [
            numpy
            torch
            torchaudio
            transformers
            peft
            safetensors
            librosa
            rich
            click
            loguru
            fastapi
            uvicorn
            pydantic
            pysilero-vad
            strops
            websockets
          ]
        ))

        # Dev tools
        basedpyright
        ruff
        pre-commit
      ];

      shellHook = ''
        export PS1="$(echo -e '\uf3e2') {\[$(tput sgr0)\]\[\033[38;5;228m\]\w\[$(tput sgr0)\]\[\033[38;5;15m\]} (${name}) \\$ \[$(tput sgr0)\]"
        export PYTHONPATH="$(pwd):$PYTHONPATH"
      '';
    };

    packages = { };
  };
}

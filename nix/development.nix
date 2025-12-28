{ inputs, ... }:

let
  self = inputs.self;
  nixpkgs = inputs.nixpkgs;
in {
  flake.overlays.dev = nixpkgs.lib.composeManyExtensions [
    (final: prev: {
      # Add custom packages here, for example:
      # my-package = final.callPackage ./pkgs/my-package {};

      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (py-final: py-prev: {
          fiddle = py-final.callPackage ./pkgs/fiddle {};
          nv-one-logger-core = py-final.callPackage ./pkgs/nv-one-logger-core {};
          nemo-toolkit = py-final.callPackage ./pkgs/nemo-toolkit {};
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
            nemo-toolkit
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

    packages = {
      inherit (pkgs-dev.python3Packages) nemo-toolkit nv-one-logger-core fiddle;
    };
  };
}

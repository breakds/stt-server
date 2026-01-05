{ inputs, ... }:

let
  inherit (inputs) self nixpkgs crane advisory-db;
in {
  perSystem = { system, pkgs-strops, lib, ... }: let
    craneLib = crane.mkLib pkgs-strops;

    src = craneLib.cleanCargoSource ../.;

    commonArgs = {
      inherit src;
      strictDeps = true;
      buildInputs = [
        # Add additional build inputs here
      ] ++ lib.optionals pkgs-strops.stdenv.isDarwin [
        pkgs-strops.libiconv
      ];
    };

    cargoArtifacts = craneLib.buildDepsOnly commonArgs;

  in {
    _module.args.pkgs-strops = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
      };
    };

    checks = {
      strops-clippy = craneLib.cargoClippy (commonArgs // { inherit cargoArtifacts; });

      strops-doc = craneLib.cargoDoc (commonArgs // {
        inherit cargoArtifacts;
        env.RUSTDOCFLAGS = "--deny warnings";
      });

      strops-fmt = craneLib.cargoFmt {
        inherit src;
      };

      strops-tool-fmt = craneLib.taploFmt {
        src = lib.sources.sourceFilesBySuffices src [ ".toml" ];
      };

      strops-deny = craneLib.cargoDeny {
        inherit src;
      };

      strops-nextest = craneLib.cargoNextest (commonArgs // {
        inherit cargoArtifacts;
        partitions = 1;
        partitionType = "count";
        cargoNextestPartitionsExtraArgs = "--no-tests=pass";
      });
    };

    devShells.strops = let
      pythonEnv = pkgs-strops.python3.withPackages (ps: with ps; [
        # Add python dependencies here
      ]);
    in craneLib.devShell {
      checks = self.checks."${system}";
      packages = with pkgs-strops; [
        pythonEnv
        maturin
      ];
    };
  };
}

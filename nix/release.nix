# Release configuration for stt-server
#
# This file exports:
# - overlays.default: Complete overlay with all dependencies and stt-server
# - nixosModules.default: NixOS service module for stt-server
#
# Usage in your flake.nix:
#   inputs.stt-server.url = "github:user/stt-server";
#
#   # Add package to your nixpkgs
#   nixpkgs.overlays = [ stt-server.overlays.default ];
#
#   # Use the NixOS module
#   imports = [ stt-server.nixosModules.default ];
{ inputs, ... }:

let
  inherit (inputs) self nixpkgs;
in
{
  # Complete overlay that includes all dependencies (ml-pkgs, strops) and stt-server
  flake.overlays.default = nixpkgs.lib.composeManyExtensions [
    # Include the dev overlay (ml-pkgs + strops)
    self.overlays.dev
    # Add stt-server package
    (final: prev: {
      pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
        (py-final: py-prev: {
          stt-server = final.callPackage ./pkgs/stt-server/package.nix {};
        })
      ];
      stt-server = with final.python3Packages; toPythonApplication stt-server;
    })
  ];

  # NixOS module for the service
  flake.nixosModules.default = import ./modules/stt-server.nix;
  flake.nixosModules.stt-server = import ./modules/stt-server.nix;

  # Expose the package in flake outputs
  perSystem = { system, pkgs, ... }: {
    _module.args.pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
        cudaForwardCompat = true;
        cudaCapabilities = [ "7.5" "8.6" "8.9" "12.0" ];
      };
      overlays = [ self.overlays.default ];
    };
    packages.default = pkgs.stt-server;
  };
}

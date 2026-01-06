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
      stt-server = final.callPackage ./packages/stt-server.nix {};
    })
  ];

  # NixOS module for the service
  flake.nixosModules.default = import ./modules/stt-server.nix;
  flake.nixosModules.stt-server = import ./modules/stt-server.nix;

  # Expose the package in flake outputs
  # Uses pkgs-dev which has all required overlays applied
  perSystem = { system, pkgs-dev, ... }: {
    packages.stt-server = pkgs-dev.callPackage ./packages/stt-server.nix {};
  };
}

{
  stdenv,
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  pytestCheckHook,
  pythonOlder,
  rustPlatform,
}:

buildPythonPackage rec {
  pname = "strops";
  version = "0.5.0";
  format = "pyproject";

  src = lib.cleanSourceWith {
    filter = name: type: ! (( type == "regular" ) && lib.hasSuffix ".nix" (baseNameOf name));
    src = lib.cleanSource ../.;
  };

  cargoDeps = rustPlatform.fetchCargoVendor {
    inherit pname version src;
    hash = lib.fakeHash;
  };

  nativeBuildInputs = with rustPlatform; [
    cargoSetupHook
    maturinBuildHook
  ];

  propagatedBuildInputs = [];

  pythonImportsCheck = [ "strops" ];
}

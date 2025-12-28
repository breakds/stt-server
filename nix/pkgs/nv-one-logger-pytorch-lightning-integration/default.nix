{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  poetry-core,
  pythonRelaxDepsHook,
  strenum,
  lightning,
  setuptools,
  nv-one-logger-core,
  nv-one-logger-training-telemetry,
}:

buildPythonPackage rec {
  pname = "nv-one-logger-pytorch-lightning-integration";
  version = "2.3.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "nv-one-logger";
    tag = "nv-one-logger-pytorch-lightning-integration-v${version}";
    hash = "sha256-9ofGv/t9Y4KlnxK/sB/Qu3NIiod01W8ITVWEpfHEoI8=";
  };

  sourceRoot = "${src.name}/nv_one_logger/one_logger_pytorch_lightning_integration";

  build-system = [ poetry-core ];

  nativeBuildInputs = [ pythonRelaxDepsHook ];

  # python-semantic-release is only used for CI versioning, not needed for building
  # Also relax poetry-core version constraint and add PEP 621 [project] section for poetry-core 2.x
  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace-fail 'requires = ["poetry-core>=1.9.1,<2.0.0", "python-semantic-release>=9.8.3,<10.0"]' \
                     'requires = ["poetry-core"]'

    # Add PEP 621 [project] section required by poetry-core 2.x
    cat >> pyproject.toml << 'EOF'

[project]
name = "nv-one-logger-pytorch-lightning-integration"
version = "2.3.0"
requires-python = ">=3.9"
EOF
  '';

  dependencies = [
    strenum
    lightning
    setuptools
    nv-one-logger-core
    nv-one-logger-training-telemetry
  ];

  pythonRelaxDeps = [
    "StrEnum"
    "lightning"
    "setuptools"
    "nv-one-logger-core"
    "nv-one-logger-training-telemetry"
  ];

  # Tests require additional setup
  doCheck = false;

  pythonImportsCheck = [ "nv_one_logger.training_telemetry.integration" ];

  meta = {
    description = "NVIDIA One Logger PyTorch Lightning Integration - training telemetry for PyTorch Lightning";
    homepage = "https://github.com/NVIDIA/nv-one-logger";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ ];
    platforms = lib.platforms.linux;
  };
}

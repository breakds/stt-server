{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  poetry-core,
  pythonRelaxDepsHook,
  strenum,
  typing-extensions,
  nv-one-logger-core,
}:

buildPythonPackage rec {
  pname = "nv-one-logger-training-telemetry";
  version = "2.3.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "nv-one-logger";
    tag = "nv-one-logger-training-telemetry-v${version}";
    hash = "sha256-9ofGv/t9Y4KlnxK/sB/Qu3NIiod01W8ITVWEpfHEoI8=";
  };

  sourceRoot = "${src.name}/nv_one_logger/one_logger_training_telemetry";

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
name = "nv-one-logger-training-telemetry"
version = "2.3.0"
requires-python = ">=3.8"
EOF
  '';

  dependencies = [
    strenum
    typing-extensions
    nv-one-logger-core
  ];

  pythonRelaxDeps = [
    "StrEnum"
    "typing-extensions"
    "nv-one-logger-core"
  ];

  # Tests require additional setup
  doCheck = false;

  pythonImportsCheck = [ "nv_one_logger.training_telemetry" ];

  meta = {
    description = "NVIDIA One Logger Training Telemetry - training job telemetry utilities";
    homepage = "https://github.com/NVIDIA/nv-one-logger";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ ];
    platforms = lib.platforms.linux;
  };
}

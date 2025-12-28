{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  poetry-core,
  pythonRelaxDepsHook,
  pydantic,
  overrides,
  strenum,
  toml,
  typing-extensions,
}:

buildPythonPackage rec {
  pname = "nv-one-logger-core";
  version = "2.3.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "nv-one-logger";
    tag = "nv-one-logger-core-v${version}";
    hash = "sha256-9ofGv/t9Y4KlnxK/sB/Qu3NIiod01W8ITVWEpfHEoI8=";
  };

  sourceRoot = "${src.name}/nv_one_logger/one_logger_core";

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
name = "nv-one-logger-core"
version = "2.3.0"
requires-python = ">=3.8"
EOF
  '';

  dependencies = [
    pydantic
    overrides
    strenum
    toml
    typing-extensions
  ];

  pythonRelaxDeps = [
    "pydantic"
    "overrides"
    "StrEnum"
    "toml"
    "typing-extensions"
  ];

  # Tests require additional setup
  doCheck = false;

  pythonImportsCheck = [ "nv_one_logger" ];

  meta = {
    description = "NVIDIA One Logger Core - logging utilities for NVIDIA AI frameworks";
    homepage = "https://github.com/NVIDIA/nv-one-logger";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ ];
    platforms = lib.platforms.linux;
  };
}

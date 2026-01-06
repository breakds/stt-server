# Nix package for stt-server
#
# This creates a Python application that runs the STT WebSocket server.
# The package includes all dependencies and provides the `stt-server` command.
{ lib,
  buildPythonPackage,
  hatchling,
  numpy,
  torch,
  torchaudio,
  transformers,
  peft,
  safetensors,
  librosa,
  fastapi,
  uvicorn,
  pydantic,
  loguru,
  click,
  websockets,
  rich,
  pysilero-vad,
  strops,
}:


buildPythonPackage {
  pname = "stt-server";
  version = "0.8.0";
  pyproject = true;

  src = lib.cleanSourceWith {
    filter = name: type:
      let baseName = baseNameOf name;
      in !(
        # Exclude development/test files
        baseName == ".git" ||
        baseName == ".direnv" ||
        baseName == ".envrc" ||
        baseName == "__pycache__" ||
        baseName == ".pytest_cache" ||
        baseName == "exploration" ||
        baseName == "examples" ||
        lib.hasSuffix ".pyc" baseName ||
        lib.hasSuffix ".nix" baseName
      );
    src = lib.cleanSource ../../..;
  };

  build-system = [ hatchling ];

  dependencies = [
    numpy
    torch
    torchaudio
    transformers
    peft
    safetensors
    librosa
    fastapi
    uvicorn
    pydantic
    loguru
    click
    websockets
    rich
    pysilero-vad
    strops
  ];

  passthru = {
    inherit (torch) cudaSupport cudaCapabilities;
  };

  meta = with lib; {
    description = "Real-time Speech-to-Text WebSocket server for conversational agents";
    homepage = "https://github.com/breakds/stt-server/";
    license = licenses.mit;
    mainProgram = "stt-server";
    maintainers = with maintainers; [ breakds ];
  };

}

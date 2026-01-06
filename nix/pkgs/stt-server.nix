# Nix package for stt-server
#
# This creates a Python application that runs the STT WebSocket server.
# The package includes all dependencies and provides the `stt-server` command.
{ lib
, python3
, writeShellApplication
}:

let
  pythonEnv = python3.withPackages (ps: with ps; [
    # Core runtime
    numpy
    torch
    torchaudio
    transformers
    peft
    safetensors
    librosa

    # Server
    fastapi
    uvicorn
    pydantic
    websockets
    loguru
    click

    # VAD
    pysilero-vad

    # Custom packages (must be in overlay)
    strops
  ]);

  # The source directory containing the Python packages
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
    src = lib.cleanSource ../..;
  };
in
writeShellApplication {
  name = "stt-server";

  runtimeInputs = [ pythonEnv ];

  text = ''
    export PYTHONPATH="${src}:''${PYTHONPATH:-}"
    exec python -m stt_server.server "$@"
  '';

  meta = with lib; {
    description = "Real-time Speech-to-Text WebSocket server for conversational agents";
    homepage = "https://github.com/user/stt-server";
    license = licenses.mit;
    mainProgram = "stt-server";
  };
}

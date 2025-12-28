{
  lib,
  buildPythonPackage,
  fetchFromGitHub,
  pythonRelaxDepsHook,
  setuptools,
  # Core dependencies (requirements.txt)
  numpy,
  torch,
  tqdm,
  wrapt,
  ruamel-yaml,
  scikit-learn,
  tensorboard,
  onnx,
  huggingface-hub,
  python-dateutil,
  wget,
  text-unidecode,
  numba,
  numexpr,
  protobuf,
  fsspec,
  # Lightning dependencies (requirements_lightning.txt)
  pytorch-lightning,
  lightning,
  hydra-core,
  omegaconf,
  transformers,
  webdataset,
  torchmetrics,
  cloudpickle,
  peft,
  wandb,
  fiddle,
  nv-one-logger-core,
  nv-one-logger-training-telemetry,
  nv-one-logger-pytorch-lightning-integration,
  # Common dependencies (requirements_common.txt)
  einops,
  pandas,
  inflect,
  sentencepiece,
  datasets,
  sacremoses,
  # ASR dependencies (requirements_asr.txt)
  librosa,
  soundfile,
  scipy,
  packaging,
  editdistance,
  braceexpand,
  pydub,
  resampy,
  marshmallow,
  optuna,
  # Extra useful deps
  sacrebleu,
  rouge-score,
  nltk,
}:

buildPythonPackage rec {
  pname = "nemo-toolkit";
  version = "2.6.0";
  pyproject = true;

  src = fetchFromGitHub {
    owner = "NVIDIA";
    repo = "NeMo";
    tag = "v${version}";
    hash = "sha256-JgzJzGq7nWbFm+y4A1TWEl5ps41PbqQjiEo4VV4kX3M=";
  };

  build-system = [ setuptools ];

  nativeBuildInputs = [ pythonRelaxDepsHook ];

  dependencies = [
    # Core (requirements.txt)
    numpy
    torch
    tqdm
    wrapt
    ruamel-yaml
    scikit-learn
    tensorboard
    onnx
    huggingface-hub
    python-dateutil
    wget
    text-unidecode
    numba
    numexpr
    protobuf
    fsspec
    # Lightning (requirements_lightning.txt)
    pytorch-lightning
    lightning
    hydra-core
    omegaconf
    transformers
    webdataset
    torchmetrics
    cloudpickle
    peft
    wandb
    fiddle
    nv-one-logger-core
    nv-one-logger-training-telemetry
    nv-one-logger-pytorch-lightning-integration
    # Common (requirements_common.txt)
    einops
    pandas
    inflect
    sentencepiece
    datasets
    sacremoses
    # ASR (requirements_asr.txt)
    librosa
    soundfile
    scipy
    packaging
    editdistance
    braceexpand
    pydub
    resampy
    marshmallow
    optuna
    # Extra useful deps
    sacrebleu
    rouge-score
    nltk
  ];

  pythonRelaxDeps = [
    "numpy"
    "protobuf"
    "numexpr"
    "lightning"
    "fsspec"
    "huggingface-hub"
    "sentencepiece"
    "librosa"
  ];

  # Remove dependencies that are not packaged in nixpkgs or not needed for ASR
  pythonRemoveDeps = [
    # ASR deps not in nixpkgs
    "lhotse"
    "kaldialign"
    "ctc-segmentation"
    "kaldi-python-io"
    "jiwer"
    "sox"
    "pyannote.core"
    "pyannote.metrics"
    "pyloudnorm"
    "whisper-normalizer"
    # Common deps not in nixpkgs
    "mediapy"
    # CUDA-specific (handled by torch)
    "numba-cuda"
  ];

  # Disable tests as they require GPU and additional fixtures
  doCheck = false;

  pythonImportsCheck = [ "nemo" ];

  meta = {
    description = "NVIDIA NeMo - a toolkit for conversational AI";
    homepage = "https://github.com/NVIDIA/NeMo";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ ];
    platforms = lib.platforms.linux;
  };
}

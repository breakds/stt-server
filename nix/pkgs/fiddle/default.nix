{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
  flit-core,
  absl-py,
  graphviz,
  libcst,
  typing-extensions,
}:

buildPythonPackage rec {
  pname = "fiddle";
  version = "0.3.0";
  pyproject = true;

  src = fetchPypi {
    inherit pname version;
    hash = "sha256-XQg9MpmkeYaDRVEzhabFVGFBvZIIbBXT3L+ACKkAddM=";
  };

  build-system = [
    setuptools
    flit-core
  ];

  dependencies = [
    absl-py
    graphviz
    libcst
    typing-extensions
  ];

  # Tests require additional dependencies not available
  doCheck = false;

  pythonImportsCheck = [ "fiddle" ];

  meta = {
    description = "A Python-first configuration library particularly well suited to ML applications";
    homepage = "https://github.com/google/fiddle";
    license = lib.licenses.asl20;
    maintainers = with lib.maintainers; [ ];
  };
}

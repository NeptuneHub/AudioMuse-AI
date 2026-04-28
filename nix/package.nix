{
  lib,
  stdenvNoCC,
  fetchurl,
  python312,
  python312Packages,
  makeWrapper,
  models,
  ffmpeg,
  fftw,
  libsndfile,
  libsamplerate,
  openblas,
  lapack,
  postgresql,
}:

let
  python = python312;

  pythonEnv = python.withPackages (
    ps: with ps; [
      numpy
      scipy
      numba
      soundfile
      flask
      flask-cors
      redis
      requests
      scikit-learn
      rq
      pyyaml
      six
      rapidfuzz
      psycopg2
      ftfy
      flasgger
      sqlglot
      google-genai
      pydub
      psutil
      onnx
      onnxruntime
      resampy
      librosa
      mutagen
      flatbuffers
      packaging
      protobuf
      sympy
      mcp
      httpx
      transformers
      sentencepiece
      pyjwt
      argon2-cffi
      gunicorn
      zstandard
      umap-learn
    ]
  );

  # Pre-fetched wheels for packages not in nixpkgs
  python-mpd2-wheel = fetchurl {
    url = "https://files.pythonhosted.org/packages/8e/6d/1b9e1c203057c9a7fa6971db3605188a8ef1120ca305e4878c960ab6e2d3/python_mpd2-3.1.1-py2.py3-none-any.whl";
    hash = "sha256-hr8RAKCxNZWddKmnpYzwUVvzC7VOslrm+44XXlAwD8M=";
  };

  voyager-wheel = fetchurl {
    url = "https://files.pythonhosted.org/packages/b7/13/a772a9a2d4cc427f6b4ae2aca65e50ec99f7bb6037c346cc22a07bc4a326/voyager-2.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl";
    hash = "sha256-lBW5ySO6xJPVxo1VrlXEKJFTwj+fR4Qik03mGsuvTAE=";
  };

  laion-clap-wheel = fetchurl {
    url = "https://files.pythonhosted.org/packages/ab/ea/6fb3d90bc1d85e6039506c248009e7f195456b81efa0a7aa413b3e13eb04/laion_clap-1.1.7-py3-none-any.whl";
    hash = "sha256-VFECT7lWOm94GUVkgVHSNChFwOBu+5MExKngiuVxEgk=";
  };

  # nixpkgs mistralai is v2.x but project needs <2.0.0
  mistralai-wheel = fetchurl {
    url = "https://files.pythonhosted.org/packages/f9/26/71cca7ceb9d5956511a560c98ba48562bf45ab6dd4dc0a026d2298ee60cf/mistralai-1.11.1-py3-none-any.whl";
    hash = "sha256-w2LM2IQESL89unyI69loPQdvAogpjdh6UeTL3ubq+sE=";
  };

  # pozalabs-pydub is a fork of pydub fixing a Python 3.12 SyntaxWarning.
  # nixpkgs pydub (already in pythonEnv) works fine — the warning is cosmetic.

  # Packages not in nixpkgs — installed from pre-fetched wheels
  pipVendor = stdenvNoCC.mkDerivation {
    pname = "audiomuse-ai-pip-vendor";
    version = "0.1.0";

    dontUnpack = true;

    nativeBuildInputs = [
      python
      python312Packages.pip
      python312Packages.setuptools
      python312Packages.wheel
    ];

    buildPhase = ''
      export HOME=$TMPDIR
      mkdir -p $out/lib/${python.libPrefix}/site-packages

      # Create a directory with properly-named wheels (strip Nix store hash prefix)
      mkdir -p wheels
      cp ${python-mpd2-wheel} wheels/python_mpd2-3.1.1-py2.py3-none-any.whl
      cp ${voyager-wheel} wheels/voyager-2.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
      cp ${laion-clap-wheel} wheels/laion_clap-1.1.7-py3-none-any.whl
      cp ${mistralai-wheel} wheels/mistralai-1.11.1-py3-none-any.whl

      # Install from local wheel directory (no network needed)
      ${python}/bin/python -m pip install \
        --no-deps \
        --no-index \
        --find-links wheels/ \
        --target $out/lib/${python.libPrefix}/site-packages \
        --no-cache-dir \
        python-mpd2 \
        voyager==2.1.0 \
        laion-clap \
        mistralai==1.11.1
    '';

    installPhase = "true";

    meta.description = "Pip-vendored Python packages for AudioMuse-AI";
  };

  appSrc = lib.cleanSourceWith {
    src = ./..;
    filter =
      path: type:
      let
        baseName = baseNameOf path;
        relPath = lib.removePrefix (toString ./..) path;
      in
      # Include Python files at root
      (type == "regular" && lib.hasSuffix ".py" baseName && !lib.hasPrefix "/nix" relPath)
      # Include key directories
      || (
        type == "directory"
        && builtins.elem baseName [
          "templates"
          "static"
          "tasks"
          "query"
          "student_clap"
          "requirements"
        ]
      )
      # Include everything inside included directories
      || (
        lib.hasPrefix "/templates" relPath
        || lib.hasPrefix "/static" relPath
        || lib.hasPrefix "/tasks" relPath
        || lib.hasPrefix "/query" relPath
        || lib.hasPrefix "/student_clap" relPath
      )
      # Include JSON data file
      || (type == "regular" && baseName == "mood_centroids_real_080_clap.json");
  };

in
stdenvNoCC.mkDerivation {
  pname = "audiomuse-ai";
  version = "1.0.4";

  src = appSrc;

  nativeBuildInputs = [ makeWrapper ];

  buildInputs = [
    ffmpeg
    fftw
    libsndfile
    libsamplerate
    openblas
    lapack
    postgresql.lib
  ];

  installPhase = ''
    runHook preInstall

    # Copy application source
    mkdir -p $out/lib/audiomuse-ai
    cp -r . $out/lib/audiomuse-ai/

    # Create wrapper scripts
    mkdir -p $out/bin

    local commonArgs=(
      --prefix PYTHONPATH : "${pythonEnv}/${python.sitePackages}"
      --prefix PYTHONPATH : "${pipVendor}/lib/${python.libPrefix}/site-packages"
      --prefix PYTHONPATH : "$out/lib/audiomuse-ai"
      --prefix PATH : "${lib.makeBinPath [ ffmpeg ]}"
      --set EMBEDDING_MODEL_PATH "${models}/model/musicnn_embedding.onnx"
      --set PREDICTION_MODEL_PATH "${models}/model/musicnn_prediction.onnx"
      --set CLAP_AUDIO_MODEL_PATH "${models}/model/model_epoch_36.onnx"
      --set CLAP_TEXT_MODEL_PATH "${models}/model/clap_text_model.onnx"
      --set HF_HOME "${models}/cache/huggingface"
      --set HF_HUB_OFFLINE "1"
      --set TRANSFORMERS_OFFLINE "1"
    )

    # Flask web server (port passed at runtime via --bind)
    makeWrapper ${pythonEnv}/bin/gunicorn $out/bin/audiomuse-ai-flask \
      "''${commonArgs[@]}" \
      --chdir "$out/lib/audiomuse-ai" \
      --add-flags "--workers 1 --threads 4 --worker-class gthread --keep-alive 5 --timeout 300 app:app"

    # Default queue worker
    makeWrapper ${pythonEnv}/bin/python $out/bin/audiomuse-ai-worker-default \
      "''${commonArgs[@]}" \
      --set AUDIOMUSE_ROLE "worker" \
      --chdir "$out/lib/audiomuse-ai" \
      --add-flags "$out/lib/audiomuse-ai/rq_worker.py"

    # High priority queue worker
    makeWrapper ${pythonEnv}/bin/python $out/bin/audiomuse-ai-worker-high \
      "''${commonArgs[@]}" \
      --set AUDIOMUSE_ROLE "worker" \
      --chdir "$out/lib/audiomuse-ai" \
      --add-flags "$out/lib/audiomuse-ai/rq_worker_high_priority.py"

    # Janitor
    makeWrapper ${pythonEnv}/bin/python $out/bin/audiomuse-ai-janitor \
      "''${commonArgs[@]}" \
      --set AUDIOMUSE_ROLE "worker" \
      --chdir "$out/lib/audiomuse-ai" \
      --add-flags "$out/lib/audiomuse-ai/rq_janitor.py"

    runHook postInstall
  '';

  meta = with lib; {
    description = "AI-powered music analysis and playlist generation";
    homepage = "https://github.com/NeptuneHub/AudioMuse-AI";
    license = licenses.gpl3;
    platforms = platforms.linux;
    mainProgram = "audiomuse-ai-flask";
  };
}

{
  lib,
  stdenvNoCC,
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
      mistralai
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

  # Packages not in nixpkgs — installed via pip into a vendor directory
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

      # Install packages not available in nixpkgs
      ${python}/bin/python -m pip install \
        --no-deps \
        --target $out/lib/${python.libPrefix}/site-packages \
        --no-cache-dir \
        python-mpd2 \
        pozalabs-pydub==0.37.0 \
        voyager==2.1.0 \
        laion-clap
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

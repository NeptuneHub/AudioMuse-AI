{
  lib,
  stdenvNoCC,
  fetchurl,
}:

let
  baseUrl = "https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v4.0.0-model";
  dclapUrl = "https://github.com/NeptuneHub/AudioMuse-AI-DCLAP/releases/download/v1";
in
stdenvNoCC.mkDerivation {
  pname = "audiomuse-ai-models";
  version = "4.0.0";

  srcs = [
    (fetchurl {
      url = "${baseUrl}/musicnn_embedding.onnx";
      hash = "sha256-pIrYh5UKVXrvu03N31itSAIhP/X8b7Ue2jUHzXl7ubA=";
      name = "musicnn_embedding.onnx";
    })
    (fetchurl {
      url = "${baseUrl}/musicnn_prediction.onnx";
      hash = "sha256-DU543UPGEK7IjAmeQfSolpeX2lrCYS6myiH6qeGkKPM=";
      name = "musicnn_prediction.onnx";
    })
    (fetchurl {
      url = "${dclapUrl}/model_epoch_36.onnx";
      hash = "sha256-F4YEA/j8kK/4rAYyoHQeteWNjAsK0vzlztlnJ0sOqXE=";
      name = "model_epoch_36.onnx";
    })
    (fetchurl {
      url = "${dclapUrl}/model_epoch_36.onnx.data";
      hash = "sha256-KnNbI8Kq17Etn/yFM0zrzGWcB2ltL/YOLjeNoott9lc=";
      name = "model_epoch_36.onnx.data";
    })
    (fetchurl {
      url = "${baseUrl}/clap_text_model.onnx";
      hash = "sha256-IA1I85Bf8fJyr1AG3ZhR+UBxp93k6v2cB7wJxaxlpxQ=";
      name = "clap_text_model.onnx";
    })
  ];

  huggingfaceModels = fetchurl {
    url = "${baseUrl}/huggingface_models.tar.gz";
    hash = lib.fakeHash;
    name = "huggingface_models.tar.gz";
  };

  dontUnpack = true;

  installPhase = ''
    runHook preInstall

    mkdir -p $out/model $out/cache/huggingface

    for src in $srcs; do
      # Strip the hash prefix from the filename
      local fname=$(basename "$src")
      fname=''${fname#*-}
      cp "$src" "$out/model/$fname"
    done

    tar -xzf "$huggingfaceModels" -C $out/cache/huggingface

    runHook postInstall
  '';

  meta = with lib; {
    description = "Pre-trained ML models for AudioMuse-AI";
    homepage = "https://github.com/NeptuneHub/AudioMuse-AI";
    license = licenses.free;
    platforms = platforms.all;
  };
}

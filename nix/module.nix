{
  config,
  lib,
  ...
}:

let
  inherit (lib)
    mkEnableOption
    mkOption
    mkIf
    mkDefault
    types
    optional
    literalExpression
    ;
  cfg = config.services.audiomuse-ai;

  # Build the common environment variables shared by all services
  commonEnv = {
    POSTGRES_HOST = cfg.postgresql.host;
    POSTGRES_PORT = toString cfg.postgresql.port;
    POSTGRES_USER = cfg.postgresql.user;
    POSTGRES_DB = cfg.postgresql.database;
    REDIS_URL = "redis://${cfg.redis.host}:${toString cfg.redis.port}/0";
    TEMP_DIR = cfg.tempDir;
    MEDIASERVER_TYPE = cfg.mediaServer.type;
    AI_MODEL_PROVIDER = cfg.aiModelProvider;
    PYTHONUNBUFFERED = "1";
  }
  // lib.optionalAttrs (cfg.mediaServer.url != "") {
    "${lib.toUpper cfg.mediaServer.type}_URL" = cfg.mediaServer.url;
  }
  // lib.optionalAttrs cfg.gpu.enable {
    USE_GPU_CLUSTERING = "true";
  }
  // cfg.extraEnvironment;

  # Common systemd service configuration
  commonServiceConfig = {
    User = "audiomuse-ai";
    Group = "audiomuse-ai";
    StateDirectory = "audiomuse-ai";
    WorkingDirectory = "${cfg.package}/lib/audiomuse-ai";
    Restart = "always";
    RestartSec = 5;

    # Hardening
    NoNewPrivileges = true;
    ProtectSystem = "strict";
    ProtectHome = true;
    PrivateTmp = true;
    ReadWritePaths = [
      cfg.tempDir
      "/var/lib/audiomuse-ai"
    ];
  }
  // lib.optionalAttrs (cfg.environmentFile != null) {
    EnvironmentFile = cfg.environmentFile;
  };

  # Service dependencies
  serviceDeps =
    optional cfg.postgresql.createLocally "audiomuse-ai-db-setup.service"
    ++ optional cfg.redis.createLocally "redis-audiomuse-ai.service";

in
{
  options.services.audiomuse-ai = {
    enable = mkEnableOption "AudioMuse-AI music analysis service";

    package = mkOption {
      type = types.package;
      description = "The AudioMuse-AI package to use.";
    };

    port = mkOption {
      type = types.port;
      default = 8000;
      description = "Port for the Flask web server.";
    };

    # --- PostgreSQL ---
    postgresql = {
      host = mkOption {
        type = types.str;
        default = "localhost";
        description = "PostgreSQL host.";
      };

      port = mkOption {
        type = types.port;
        default = 5432;
        description = "PostgreSQL port.";
      };

      user = mkOption {
        type = types.str;
        default = "audiomuse";
        description = "PostgreSQL user.";
      };

      database = mkOption {
        type = types.str;
        default = "audiomusedb";
        description = "PostgreSQL database name.";
      };

      createLocally = mkOption {
        type = types.bool;
        default = true;
        description = "Whether to create the PostgreSQL database and user locally.";
      };
    };

    # --- Redis ---
    redis = {
      host = mkOption {
        type = types.str;
        default = "localhost";
        description = "Redis host.";
      };

      port = mkOption {
        type = types.port;
        default = 6379;
        description = "Redis port.";
      };

      createLocally = mkOption {
        type = types.bool;
        default = true;
        description = "Whether to create a local Redis instance.";
      };
    };

    # --- Secrets ---
    environmentFile = mkOption {
      type = types.nullOr types.path;
      default = null;
      description = ''
        Path to an environment file containing secrets.
        This file is loaded by all AudioMuse-AI systemd services via EnvironmentFile=.
        It should contain lines like:
          POSTGRES_PASSWORD=mysecretpassword
          JELLYFIN_TOKEN=mytoken
          JWT_SECRET=myjwtsecret
          AUDIOMUSE_USER=admin
          AUDIOMUSE_PASSWORD=adminpass
          API_TOKEN=myapitoken
          OPENAI_API_KEY=sk-...
          GEMINI_API_KEY=AIza...
          MISTRAL_API_KEY=...
          NAVIDROME_PASSWORD=...
      '';
    };

    # --- Media Server ---
    mediaServer = {
      type = mkOption {
        type = types.enum [
          "jellyfin"
          "navidrome"
          "lyrion"
          "mpd"
          "emby"
        ];
        default = "jellyfin";
        description = "Type of media server to connect to.";
      };

      url = mkOption {
        type = types.str;
        default = "";
        description = "URL of the media server. Secrets (tokens, passwords) should go in environmentFile.";
      };
    };

    # --- AI ---
    aiModelProvider = mkOption {
      type = types.enum [
        "NONE"
        "OLLAMA"
        "OPENAI"
        "GEMINI"
        "MISTRAL"
      ];
      default = "NONE";
      description = "AI model provider for playlist naming.";
    };

    # --- GPU ---
    gpu.enable = mkOption {
      type = types.bool;
      default = false;
      description = "Enable GPU acceleration (requires NVIDIA GPU and CUDA).";
    };

    # --- Workers ---
    workers = {
      default = mkOption {
        type = types.ints.positive;
        default = 1;
        description = "Number of default queue workers (song analysis).";
      };

      highPriority = mkOption {
        type = types.ints.positive;
        default = 1;
        description = "Number of high priority queue workers.";
      };
    };

    # --- Temp directory ---
    tempDir = mkOption {
      type = types.str;
      default = "/var/lib/audiomuse-ai/temp_audio";
      description = "Directory for temporary audio files.";
    };

    # --- Extra environment ---
    extraEnvironment = mkOption {
      type = types.attrsOf types.str;
      default = { };
      description = "Additional environment variables for all AudioMuse-AI services.";
      example = literalExpression ''
        {
          TZ = "Europe/Berlin";
          CLUSTER_ALGORITHM = "kmeans";
        }
      '';
    };
  };

  config = mkIf cfg.enable {
    # --- System user ---
    users.users.audiomuse-ai = {
      isSystemUser = true;
      group = "audiomuse-ai";
      home = "/var/lib/audiomuse-ai";
      createHome = true;
      description = "AudioMuse-AI service user";
    };

    users.groups.audiomuse-ai = { };

    # --- Temp directory ---
    systemd.tmpfiles.rules = [
      "d ${cfg.tempDir} 0750 audiomuse-ai audiomuse-ai -"
    ];

    # --- PostgreSQL (local) ---
    services.postgresql = mkIf cfg.postgresql.createLocally {
      enable = mkDefault true;
      ensureDatabases = [ cfg.postgresql.database ];
      ensureUsers = [
        {
          name = cfg.postgresql.user;
        }
      ];
    };

    # Grant database ownership after PostgreSQL starts
    systemd.services.audiomuse-ai-db-setup = mkIf cfg.postgresql.createLocally {
      description = "AudioMuse-AI database ownership setup";
      after = [ "postgresql.service" ];
      requires = [ "postgresql.service" ];
      wantedBy = [ "multi-user.target" ];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        User = "postgres";
        ExecStart = let
          psql = config.services.postgresql.package + "/bin/psql";
        in
          "${psql} -c \"ALTER DATABASE \\\"${cfg.postgresql.database}\\\" OWNER TO \\\"${cfg.postgresql.user}\\\"\"";
      };
    };

    # --- Redis (local) ---
    services.redis.servers.audiomuse-ai = mkIf cfg.redis.createLocally {
      enable = true;
      port = cfg.redis.port;
    };

    # --- All systemd services ---
    systemd.services = {
      # --- Flask web server ---
      audiomuse-ai-flask = {
        description = "AudioMuse-AI Flask Web Server";
        after = [ "network.target" ] ++ serviceDeps;
        requires = serviceDeps;
        wantedBy = [ "multi-user.target" ];

        environment = commonEnv // {
          SERVICE_TYPE = "flask";
        };

        serviceConfig = commonServiceConfig // {
          ExecStart = "${cfg.package}/bin/audiomuse-ai-flask --bind 0.0.0.0:${toString cfg.port}";
        };
      };

      # --- Janitor ---
      audiomuse-ai-janitor = {
        description = "AudioMuse-AI RQ Janitor";
        after = [ "network.target" ] ++ serviceDeps;
        requires = serviceDeps;
        wantedBy = [ "multi-user.target" ];

        environment = commonEnv // {
          SERVICE_TYPE = "worker";
        };

        serviceConfig = commonServiceConfig // {
          ExecStart = "${cfg.package}/bin/audiomuse-ai-janitor";
        };
      };
    }
    // lib.listToAttrs (
      # --- Default queue workers ---
      map (i: {
        name = "audiomuse-ai-worker-default-${toString i}";
        value = {
          description = "AudioMuse-AI Default Queue Worker ${toString i}";
          after = [ "network.target" ] ++ serviceDeps;
          requires = serviceDeps;
          wantedBy = [ "multi-user.target" ];

          environment = commonEnv // {
            SERVICE_TYPE = "worker";
          };

          serviceConfig = commonServiceConfig // {
            ExecStart = "${cfg.package}/bin/audiomuse-ai-worker-default";
          };
        };
      }) (lib.range 1 cfg.workers.default)
    )
    // lib.listToAttrs (
      # --- High priority queue workers ---
      map (i: {
        name = "audiomuse-ai-worker-high-${toString i}";
        value = {
          description = "AudioMuse-AI High Priority Queue Worker ${toString i}";
          after = [ "network.target" ] ++ serviceDeps;
          requires = serviceDeps;
          wantedBy = [ "multi-user.target" ];

          environment = commonEnv // {
            SERVICE_TYPE = "worker";
          };

          serviceConfig = commonServiceConfig // {
            ExecStart = "${cfg.package}/bin/audiomuse-ai-worker-high";
          };
        };
      }) (lib.range 1 cfg.workers.highPriority)
    );
  };
}

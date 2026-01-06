# NixOS service module for stt-server
#
# Provides a systemd service for running the STT WebSocket server.
#
# Example usage in configuration.nix:
#   services.stt-server = {
#     enable = true;
#     port = 15751;
#     host = "0.0.0.0";
#     device = "cuda";  # or "cpu"
#   };
{ config, lib, pkgs, ... }:

let
  cfg = config.services.stt-server;
in
{
  options.services.stt-server = {
    enable = lib.mkEnableOption "STT Server - Speech-to-Text WebSocket service";

    port = lib.mkOption {
      type = lib.types.port;
      default = 15751;
      description = "Port for the STT WebSocket server to listen on.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "0.0.0.0";
      description = "Host address to bind the server to.";
    };

    device = lib.mkOption {
      type = lib.types.enum [ "cuda" "cpu" ];
      default = "cuda";
      description = ''
        Device for running the ASR model.
        - cuda: Use NVIDIA GPU (recommended for real-time performance)
        - cpu: Use CPU (slower, but works without GPU)
      '';
    };

    package = lib.mkOption {
      type = lib.types.package;
      default = pkgs.stt-server;
      defaultText = lib.literalExpression "pkgs.stt-server";
      description = "The stt-server package to use.";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "stt-server";
      description = "User account under which stt-server runs.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "stt-server";
      description = "Group under which stt-server runs.";
    };

    cacheDir = lib.mkOption {
      type = lib.types.str;
      default = "/var/cache/stt-server";
      description = ''
        Directory for caching downloaded models (HuggingFace Hub cache).
        This avoids re-downloading models on service restart.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      home = cfg.cacheDir;
      createHome = true;
      description = "STT Server service user";
    };

    users.groups.${cfg.group} = {};

    systemd.services.stt-server = {
      description = "Speech-to-Text WebSocket Server";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      environment = {
        STT_DEVICE = cfg.device;
        HOME = cfg.cacheDir;
        XDG_CACHE_HOME = cfg.cacheDir;
        HF_HOME = "${cfg.cacheDir}/huggingface";
        TORCH_HOME = "${cfg.cacheDir}/torch";
      };

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        ExecStart = "${cfg.package}/bin/stt-server --host ${cfg.host} --port ${toString cfg.port}";
        Restart = "on-failure";
        RestartSec = "10s";

        # Hardening
        NoNewPrivileges = true;
        PrivateTmp = true;
        ProtectSystem = "strict";
        ProtectHome = true;
        ReadWritePaths = [ cfg.cacheDir ];

        # Allow GPU access when using CUDA
        SupplementaryGroups = lib.optional (cfg.device == "cuda") "video";
      };
    };

    # Ensure cache directory exists with correct permissions
    systemd.tmpfiles.rules = [
      "d ${cfg.cacheDir} 0750 ${cfg.user} ${cfg.group} -"
    ];
  };
}

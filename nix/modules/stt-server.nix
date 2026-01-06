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
  };

  config = lib.mkIf cfg.enable {
    systemd.services.stt-server = {
      description = "Speech-to-Text WebSocket Server";
      wantedBy = [ "multi-user.target" ];
      after = [ "network.target" ];

      serviceConfig = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/stt-server --host ${cfg.host} --port ${toString cfg.port}";
        Restart = "on-failure";
        RestartSec = "10s";

        # Use systemd dynamic user with persistent cache
        DynamicUser = true;
        CacheDirectory = "stt-server";

        # Environment for model caching
        Environment = [
          "STT_DEVICE=${cfg.device}"
          "HOME=/var/cache/stt-server"
          "XDG_CACHE_HOME=/var/cache/stt-server"
          "HF_HOME=/var/cache/stt-server/huggingface"
          "TORCH_HOME=/var/cache/stt-server/torch"
        ];

        # Hardening (following ollama's pattern)
        CapabilityBoundingSet = [ "" ];
        LockPersonality = true;
        NoNewPrivileges = true;
        PrivateTmp = true;
        PrivateUsers = true;
        ProtectClock = true;
        ProtectControlGroups = true;
        ProtectHome = true;
        ProtectHostname = true;
        ProtectKernelLogs = true;
        ProtectKernelModules = true;
        ProtectKernelTunables = true;
        ProtectProc = "invisible";
        ProtectSystem = "strict";
        RestrictAddressFamilies = [ "AF_INET" "AF_INET6" "AF_UNIX" ];
        RestrictNamespaces = true;
        RestrictRealtime = true;
        RestrictSUIDSGID = true;
        SystemCallArchitectures = "native";
        UMask = "0077";

        # GPU access: deny all devices by default, then allowlist NVIDIA
        PrivateDevices = cfg.device != "cuda";
      } // lib.optionalAttrs (cfg.device == "cuda") {
        DevicePolicy = "closed";
        DeviceAllow = [
          "char-nvidiactl"
          "char-nvidia-caps"
          "char-nvidia-frontend"
          "char-nvidia-uvm"
        ];
      };
    };
  };
}

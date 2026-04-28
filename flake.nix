{
  description = "AudioMuse-AI — AI-powered music analysis and playlist generation";

  inputs = {
    flake-schemas.url = "https://flakehub.com/f/DeterminateSystems/flake-schemas/*";
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.*";
  };

  outputs =
    {
      self,
      flake-schemas,
      nixpkgs,
    }:
    let
      # Packages are Linux-only (native ML libs + systemd services)
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
      forEachSupportedSystem =
        f:
        nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import nixpkgs { inherit system; };
          }
        );

      # Dev shells support more platforms
      devShellSystems = [
        "x86_64-linux"
        "aarch64-darwin"
        "x86_64-darwin"
        "aarch64-linux"
      ];
      forEachDevSystem =
        f:
        nixpkgs.lib.genAttrs devShellSystems (
          system:
          f {
            pkgs = import nixpkgs { inherit system; };
          }
        );
    in
    {
      schemas = flake-schemas.schemas;

      # Packages (Linux only)
      packages = forEachSupportedSystem (
        { pkgs }:
        let
          models = pkgs.callPackage ./nix/models.nix { };
        in
        {
          audiomuse-ai-models = models;
          default = pkgs.callPackage ./nix/package.nix { inherit models; };
        }
      );

      # NixOS module
      nixosModules.default =
        {
          config,
          lib,
          pkgs,
          ...
        }:
        {
          imports = [ ./nix/module.nix ];

          config = lib.mkIf config.services.audiomuse-ai.enable {
            services.audiomuse-ai.package = lib.mkDefault self.packages.${pkgs.system}.default;
          };
        };

      # Development environments (all platforms)
      devShells = forEachDevSystem (
        { pkgs }:
        {
          default = pkgs.mkShell {
            packages = with pkgs; [
              mkdocs
              nil
            ];
          };
        }
      );
    };
}

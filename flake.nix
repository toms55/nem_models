{
  description = "A project using NEMOSIS";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      # Define the system architecture, e.g., "x86_64-linux", "aarch64-darwin"
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # Create a Python environment with NEMOSIS and other dependencies
      pythonEnv = pkgs.python3.withPackages(p: [
        p.nemosis
        p.pandas
        p.xgboost
        p.scikit-learn
        p.matplotlib
      ]);

    in {
      # This defines a development shell (nix develop)
      devShells.${system}.default = pkgs.mkShell {
        # Packages available in the development shell
        packages = [
          pythonEnv
        ];
      };
    };
}

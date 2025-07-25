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
        # Add any other Python packages your project needs here
        # e.g., p.pandas, p.numpy, p.requests
      ]);

    in {
      # This defines a development shell (nix develop)
      devShells.${system}.default = pkgs.mkShell {
        # Packages available in the development shell
        packages = [
          pythonEnv
          # Add any other system-level tools you need in your dev environment
          # e.g., pkgs.git, pkgs.jq
        ];

        # Set up environment variables if needed
        # shellHook = ''
        #   export MY_VAR="some_value"
        # '';
      };

      # If your project has an executable script, you can define it here
      # For example, if you have a main.py script
      packages.${system}.nemosis-app = pkgs.writeShellScriptBin "nemosis-app" ''
        #!${pythonEnv}/bin/python
        # Replace 'your_script.py' with the actual entry point of your project
        # Ensure your_script.py is in the project's root or adjust the path
        exec python -m your_project_module.main "$@"
      '';

      # You might also want to define a NixOS module for system-wide installation
      # nixosModules.default = { config, lib, pkgs, ... }: {
      #   environment.systemPackages = [
      #     pythonEnv
      #   ];
      #   # Add any other system-wide configurations here
      # };
    };
}

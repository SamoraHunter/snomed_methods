# setup/install.py
"""Python-based installer for snomed_methods.

This module provides a cross-platform installation script that creates virtual
environments, installs dependencies, and registers the environment as a Jupyter
kernel. It supports both production and development installation modes.

Usage:
    python setup/install.py [venv_name] [mode]

Arguments:
    venv_name: Name of the virtual environment (default: 'snomed_methods_env')
    mode: Installation mode - 'dev' for development with extra dependencies, empty
          or any other value for production mode

Examples:
    python setup/install.py
    python setup/install.py snomed_env
    python setup/install.py snomed_env dev

"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_MIN_ARGS = 2


def run_command(cmd: str, cwd: str | Path | None = None) -> str:
    """Run a shell command and return the standard output.

    Args:
        cmd: The command string to execute.
        cwd: Optional working directory path. If provided, the command will be
            executed in this directory. Defaults to None.

    Returns:
        The stdout output from the successful command execution as a string.

    Raises:
        subprocess.CalledProcessError: If the command exits with a non-zero exit code.

    Note:
        The function uses `subprocess.run` with `check=True`, so any command failure
        will raise an exception.

    """
    result = subprocess.run(  # noqa: S603
        cmd.split(),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def get_python_executable() -> str:
    """Get the current Python executable path.

    Returns:
        The absolute path to the current Python executable as a string.

    """
    return sys.executable


def create_venv(venv_path: str | Path) -> bool:
    """Create a virtual environment at the specified path.

    Args:
        venv_path: The directory path where the virtual environment will be created.
            If the path already exists and contains a valid virtual environment,
            no action is taken.

    Returns:
        True if the virtual environment was created or already exists, False otherwise.

    Note:
        This function creates a new virtual environment using the current Python
        executable. It does not handle activation of the environment.

    """
    venv_path = Path(venv_path).resolve()

    if venv_path.exists():
        return True

    python_executable = get_python_executable()
    subprocess.run(  # noqa: S603
        [python_executable, "-m", "venv", str(venv_path)],
        capture_output=True,
        text=True,
        check=True,
        shell=False,
    )
    return True


def get_activate_script(venv_path: str | Path) -> Path:
    """Get the path to the activation script based on the current platform.

    Args:
        venv_path: The path to the virtual environment directory.

    Returns:
        A Path object pointing to the appropriate activation script:
        - Windows: venv_path/Scripts/activate.bat
        - Unix/Linux/macOS: venv_path/bin/activate

    Note:
        This function determines the correct activation script based on
        sys.platform and does not verify if the file exists.

    """
    venv_path = Path(venv_path).resolve()

    if sys.platform == "win32":
        return venv_path / "Scripts" / "activate.bat"
    return venv_path / "bin" / "activate"


def install_dependencies(venv_path: str | Path, mode: str = "production") -> None:
    """Install package dependencies in the virtual environment.

    Args:
        venv_path: The path to the virtual environment directory.
        mode: Installation mode. Use 'dev' to install development dependencies
            including extras like medcat. Any other value (including empty string)
            installs production dependencies only. Defaults to "production".

    Note:
        This function upgrades pip, setuptools, and wheel before installing the
        package. In dev mode, it installs with Extras [dev,medcat].

    """
    venv_path = Path(venv_path).resolve()

    if sys.platform == "win32":
        _pip_cmd = str(venv_path / "Scripts" / "pip.exe")
        python_cmd = str(venv_path / "Scripts" / "python.exe")
    else:
        _pip_cmd = str(venv_path / "bin" / "pip")
        python_cmd = str(venv_path / "bin" / "python")

    # Upgrade base packages
    subprocess.run(  # noqa: S603
        [python_cmd, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
        capture_output=True,
        text=True,
        check=True,
        shell=False,
    )

    # Install package
    if mode == "dev":
        subprocess.run(  # noqa: S603
            [python_cmd, "-m", "pip", "install", "-e", ".[dev,medcat]"],
            capture_output=True,
            text=True,
            check=True,
            shell=False,
        )
    else:
        subprocess.run(  # noqa: S603
            [python_cmd, "-m", "pip", "install", "-e", "."],
            capture_output=True,
            text=True,
            check=True,
            shell=False,
        )


def register_kernel(venv_name: str) -> str:
    """Register the virtual environment as a Jupyter kernel.

    Args:
        venv_name: The name to use for the kernel registration. This will be
            used as both the internal kernel name and part of the display name.

    Returns:
        The stdout output from the ipykernel install command as a string.
        If jupyter is not installed, it will be installed first (quietly).

    Note:
        This function installs jupyter if not already present and registers
        the kernel using the current Python executable's ipykernel module.
        The registered kernel will have the display name "Python (venv_name)".

    """
    python_executable = get_python_executable()

    # Install jupyter if not already installed
    subprocess.run(  # noqa: S603
        [python_executable, "-m", "pip", "install", "--quiet", "jupyter"],
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )

    # Register the kernel
    result = subprocess.run(  # noqa: S603
        [
            python_executable,
            "-m",
            "ipykernel",
            "install",
            "--user",
            "--name",
            venv_name,
            "--display-name",
            f"Python ({venv_name})",
        ],
        capture_output=True,
        text=True,
        check=True,
        shell=False,
    )

    return result.stdout


def list_kernels() -> str:
    """List all installed Jupyter kernels.

    Returns:
        A string containing the output from `jupyter kernelspec list` command.
        If the command fails, returns "Failed to list kernels".

    Note:
        This function catches CalledProcessError and returns an error message
        instead of raising an exception, making it safe for informational purposes.

    """
    python_executable = get_python_executable()

    try:
        result = subprocess.run(  # noqa: S603
            [python_executable, "-m", "jupyter", "kernelspec", "list"],
            capture_output=True,
            text=True,
            check=True,
            shell=False,
        )
    except subprocess.CalledProcessError:
        return "Failed to list kernels"
    return result.stdout


def main() -> None:
    """Main installation function that orchestrates the setup process.

    This function performs the following steps:
    1. Parses command-line arguments for venv_name and mode
    2. Creates a virtual environment at the determined path
    3. Installs package dependencies based on the specified mode
    4. Registers the environment as a Jupyter kernel

    Command-line Arguments:
        argv[1] (optional): venv_name - Name for the virtual environment.
                           Defaults to 'snomed_methods_env'
        argv[2] (optional): mode - Installation mode ('dev' or other).
                            Defaults to 'production'

    Note:
        The function does not return any value. It handles all subprocess
        calls internally and will raise exceptions if critical operations fail.

    """
    # Parse arguments
    venv_name = sys.argv[1] if len(sys.argv) > _MIN_ARGS - 1 else "snomed_methods_env"
    mode = (
        sys.argv[_MIN_ARGS - 1]
        if len(sys.argv) > _MIN_ARGS - 1 and sys.argv[_MIN_ARGS - 1] == "dev"
        else "production"
    )

    # Get paths
    script_dir = Path(__file__).parent.parent.resolve()
    venv_path = script_dir / venv_name

    # Check Python version
    get_python_executable()

    # Create virtual environment
    create_venv(venv_path)

    # Install dependencies
    install_dependencies(venv_path, mode)

    # Register kernel
    register_kernel(venv_name)

    # List kernels (platform-specific skip for older versions)
    if sys.platform == "win32":
        pass
    else:
        pass

    # Show kernel specs
    list_kernels()


if __name__ == "__main__":
    main()

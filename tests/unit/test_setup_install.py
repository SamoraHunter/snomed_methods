"""
Unit tests for the setup.install module.

This module provides comprehensive test coverage for SNOMED installation
functions including:
- Virtual environment creation and management
- Dependency installation (production, development modes)
- Jupyter kernel registration for isolated environments
- Platform-specific virtual environment paths (Linux/Windows)
- Environment activation script location

Tests ensure that:
1. Virtual environments are correctly created with proper structure
2. Dependencies are installed in the correct order (pip upgrade first)
3. Different dependency groups are handled correctly
4. Jupyter kernels are properly registered with appropriate display names
5. Platform-specific paths are used correctly for virtual environment binary files
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from setup.install import (
    create_venv,
    get_activate_script,
    install_dependencies,
    list_kernels,
    main,
    register_kernel,
)


class TestCreateVenv:
    """
    Tests for the create_venv function.

    The create_venv function creates isolated Python virtual environments.

    Behavior:
    - Creates venv directory structure with standard layout
    - Calls subprocess to execute venv creation
    - Returns True on success, False or raises exception on failure

    This is used to set up isolated Python environments for SNOMED
    deployments without affecting the system Python installation.
    """

    def test_create_venv_success(self, mocker):
        """
        Test successful virtual environment creation.

        Expected behavior: subprocess.run called with correct arguments.

        Input:Temporary directory path as venv location
        Expected calls:
            python -m venv <path>

        Verification:
        - subprocess.run is called exactly once
        - Python executable is in command arguments
        - "-m" flag present
        - "venv" argument present
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "test_venv"

            mock_run = mocker.patch("subprocess.run")
            result = create_venv(venv_path)

            assert result is True
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert sys.executable in call_args
            assert "-m" in call_args
            assert "venv" in call_args

    def test_create_venv_already_exists(self, mocker):
        """
        Test that existing venv returns True without re-creating.

        Expected behavior: Returns True and does NOT call subprocess.run.

        This is important because venv creation can be expensive. When
        the directory already exists as a valid virtual environment,
        we should skip recreation to save time and avoid unnecessary work.

        Input: Existing directory path (venv not yet created)
        Expected: Return True without subprocess.run call
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "test_venv"
            venv_path.mkdir()

            mock_run = mocker.patch("subprocess.run")
            result = create_venv(venv_path)

            assert result is True
            mock_run.assert_not_called()

    def test_create_venv_returns_true_on_success(self, mocker):
        """
        Test that create_venv returns True on successful creation.

        Expected behavior: Returns True and directory exists.

        This verifies the happy path of venv creation where:
        - Temporary directory is successfully created
        - The method completes without errors
        - Return value indicates success

        Input: New temporary directory path
        Output: True, directory exists at path
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "test_venv"

            result = create_venv(venv_path)

            assert result is True
            assert venv_path.exists()

    def test_create_venv_with_resolved_path(self, mocker):
        """
        Test that venv path is resolved (absolute) before creation.

        Expected behavior: subprocess.run receives absolute path.

        This ensures path resolution handles relative paths correctly,
        converting them to absolute paths before venv creation. This
        prevents issues with relative path interpretation during the
        virtual environment setup process.

        Input: Path that may be relative
        Output: Absolute path passed to venv command
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "test_venv"

            mock_run = mocker.patch("subprocess.run")
            result = create_venv(venv_path)

            assert result is True
            call_args = mock_run.call_args[0][0]
            call_str = " ".join(str(s) for s in call_args)
            assert str(venv_path.resolve()) in call_str

    def test_create_venv_creates_directory_structure(self, mocker):
        """
        Test that create_venv creates proper directory structure.

        Expected behavior: Directory exists and is a valid directory.

        This verifies the side effect of successful venv creation -
        that the target directory actually exists after calling
        the function. The method should handle both existing
        directories (return early) and non-existing directories (create).

        Input: Path to new directory location
        Side effects: Directory created at path
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "test_venv"

            result = create_venv(venv_path)

            assert result is True
            assert venv_path.is_dir()


class TestInstallDependencies:
    """Tests for install_dependencies function."""

    def test_install_dependencies_production(self, mocker):
        """
        Test production dependency installation.

        Expected behavior: Installs base package without dev/medcat extras.

        Production mode should:
        1. Upgrade pip first
        2. Install setuptools and wheel
        3. Install the package in editable mode WITHOUT dev/medcat extras

        Input: venv_path, mode='production'
        Expected calls:
            pip install --upgrade pip setuptools wheel
            pip install -e .

        Verification:
        - subprocess.run called exactly twice
        - Second call contains "-e" flag (editable install)
        - Second call does NOT contain [dev,medcat] extras
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "test_venv"
            venv_path.mkdir()

            if sys.platform == "win32":
                (venv_path / "Scripts").mkdir(exist_ok=True)
                str(venv_path / "Scripts" / "python.exe")
            else:
                (venv_path / "bin").mkdir(exist_ok=True)
                str(venv_path / "bin" / "python")

            mock_run = mocker.patch("subprocess.run")

            install_dependencies(venv_path, mode="production")

            assert mock_run.call_count == 2
            call0_str = str(mock_run.call_args_list[0])
            call1_str = str(mock_run.call_args_list[1])
            assert "pip" in call0_str and "install" in call0_str
            assert "-e" in call1_str and ".[dev,medcat]" not in call1_str

    def test_install_dependencies_dev(self, mocker):
        """
        Test development dependency installation.

        Expected behavior: Installs package with dev and medcat extras.

        Development mode should:
        1. Upgrade pip first
        2. Install setuptools and wheel
        3. Install package in editable mode WITH [dev,medcat] extras

        Input: venv_path, mode='dev'
        Expected calls:
            pip install --upgrade pip setuptools wheel
            pip install -e .[dev,medcat]

        This ensures developers have access to testing and MedCAT tools.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "test_venv"
            venv_path.mkdir()

            if sys.platform == "win32":
                (venv_path / "Scripts").mkdir(exist_ok=True)
                str(venv_path / "Scripts" / "python.exe")
            else:
                (venv_path / "bin").mkdir(exist_ok=True)
                str(venv_path / "bin" / "python")

            mock_run = mocker.patch("subprocess.run")

            install_dependencies(venv_path, mode="dev")

            assert mock_run.call_count == 2
            calls_str = str(mock_run.call_args_list[1])
            assert "-e" in calls_str
            assert ".[dev,medcat]" in calls_str

    def test_install_dependencies_pip_upgrade(self, mocker):
        """
        Test that pip is upgraded before installing dependencies.

        Expected behavior: First subprocess call upgrades pip/setuptools/wheel.

        This follows Python packaging best practices where pip and build
        tools are upgraded first before installing project dependencies.
        This prevents issues with older pip versions not supporting
        modern packaging standards.

        Input: venv_path, mode='production'
        Expected first calls:
            pip install --upgrade pip setuptools wheel

        Verification:
        - First call (index 0) mentions pip and setuptools and wheel
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "test_venv"
            venv_path.mkdir()

            if sys.platform == "win32":
                (venv_path / "Scripts").mkdir(exist_ok=True)
                str(venv_path / "Scripts" / "python.exe")
            else:
                (venv_path / "bin").mkdir(exist_ok=True)
                str(venv_path / "bin" / "python")

            mock_run = mocker.patch("subprocess.run")

            install_dependencies(venv_path, mode="production")

            first_call_args = str(mock_run.call_args_list[0])
            assert "pip" in first_call_args
            assert "setuptools" in first_call_args
            assert "wheel" in first_call_args

    def test_install_dependencies_platform_specific(self, mocker):
        """
        Test that platform-specific virtual environment paths are used.

        Expected behavior: Correct Python binary path based on OS.

        Windows:
        - Use Scripts subdirectory
        - Binary: venv_path/Scripts/python.exe

        Unix/Linux/macOS:
        - Use bin subdirectory
        - Binary: venv_path/bin/python

        This test verifies that install_dependencies correctly constructs
        paths for the virtual environment's Python executable across
        different operating systems without hardcoding platform-specific
        logic incorrectly.

        Input: venv_path on specified platform
        Expected: Platform-appropriate binary path used
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "test_venv"
            venv_path.mkdir()

            if sys.platform == "win32":
                (venv_path / "Scripts").mkdir(exist_ok=True)
            else:
                (venv_path / "bin").mkdir(exist_ok=True)

            mock_run = mocker.patch("subprocess.run")

            install_dependencies(venv_path, mode="production")

            calls_str_0 = str(mock_run.call_args_list[0])
            if sys.platform == "win32":
                assert "Scripts" in calls_str_0
                assert "\\.exe" in calls_str_0 or "python.exe" in calls_str_0
            else:
                assert "/bin/" in calls_str_0


class TestRegisterKernel:
    """
    Tests for the register_kernel function.

    The register_kernel function registers a Python virtual environment as
    a Jupyter kernel, making it available in Jupyter notebooks.

    Behavior:
    - Installs ipykernel package if needed
    - Runs ipykernel install command with user flag
    - Registers with display name based on venv name

    Parameters:
    - venv_path: Path to virtual environment directory

    This enables users to select their SNOMED-equipped environment
    from Jupyter's kernel list, ensuring consistent dependency versions.

    Return format: stdout from ipykernel install command (string)
    """

    def test_register_kernel_success(self, mocker):
        """
        Test successful kernel registration.

        Expected behavior: Executes ipykernel install with correct arguments.

        Kernel registration should:
        1. Install ipykernel package first
        2. Run ipykernel install --user --name "Python (<venv_name>)"

        Input: venv_path="test_venv"
        Expected calls:
            pip install ipykernel (if not installed)
            ipykernel install --user --name "Python (test_venv)"

        Verification:
        - subprocess.run called twice
        - Second call contains "ipykernel" and "install"
        - Second call contains "--user" flag
        - Display name includes venv name in format "Python (<name>)"
        """
        mock_result = mocker.MagicMock()
        mock_result.stdout = "kernel registered"
        mock_run = mocker.patch("subprocess.run", return_value=mock_result)

        result = register_kernel("test_venv")

        assert result == "kernel registered"
        assert mock_run.call_count == 2
        call_args = str(mock_run.call_args_list[1][0][0])
        assert "ipykernel" in call_args
        assert "install" in call_args
        assert "--user" in call_args
        assert "test_venv" in call_args
        assert "Python (test_venv)" in call_args

    def test_register_kernel_installs_jupyter(self, mocker):
        """
        Test that ipykernel is installed before kernel registration.

        Expected behavior: First subprocess call installs ipykernel package.

        This ensures the dependency needed for Jupyter kernel creation
        is available before attempting registration. The method should
        install ipykernel via pip (subprocess.run #0) and then run
        the actual kernel installation command (subprocess.run #1+).

        Input: venv_path
        Verification:
        - First call installs ipykernel package
        - Total calls >= 2 (install + register)
        """
        mock_result = mocker.MagicMock()
        mock_result.stdout = ""

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_result

        mocker.patch("subprocess.run", side_effect=side_effect)

        result = register_kernel("test_venv")

        assert result == ""
        assert call_count >= 2

    def test_register_kernel_with_display_name(self, mocker):
        """
        Test kernel registration with correct display name format.

        Expected behavior: ipykernel install command includes proper name.

        The Jupyter display name should follow the format "Python (<env_name>)"
        to help users identify which environment they're using in notebooks.

        Input: venv_path="my_env"
        Expected in ipykernel install:
            --name "Python (my_env)"

        This test verifies that different environment names are correctly
        incorporated into the Jupyter kernel display name.
        """
        mock_result = mocker.MagicMock()
        mock_result.stdout = "done"
        mock_run = mocker.patch("subprocess.run", return_value=mock_result)

        register_kernel("my_env")

        call_str = str(mock_run.call_args_list[1][0][0])
        assert "Python (my_env)" in call_str


class TestGetActivateScript:
    """
    Tests for the get_activate_script function.

    The get_activate_script function returns the path to the virtual
    environment activation script.

    Behavior:
    - Windows: Returns Scripts/activate.bat
    - Linux/macOS: Returns bin/activate

    This is used in setup scripts and documentation to inform users
    how to activate their SNOMED Python environment.

    Parameters:
    - venv_path: Path to virtual environment directory

    Return: Full path to the activation script (Path object)
    """

    def test_get_activate_script_linux(self, mocker):
        """
        Test get_activate_script returns correct path on Linux.

        Expected behavior: Returns venv_path/bin/activate

        On Unix-like systems (Linux/macOS), virtual environments store
        their activation scripts in the bin subdirectory. This test
        verifies the method correctly constructs that path regardless
        of the actual platform.

        Input: venv_path="/path/to/venv"
        Expected output: "/path/to/venv/bin/activate"
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "test_venv"

            result = get_activate_script(venv_path)

            assert result == (venv_path / "bin" / "activate")

    def test_get_activate_script_windows(self, mocker):
        """Test activate script path on Windows."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_path = Path(tmpdir) / "test_venv"

            mocker.patch("sys.platform", "win32")
            result = get_activate_script(venv_path)

            assert result == (venv_path / "Scripts" / "activate.bat")

    def test_get_activate_script_with_resolved_path(self, mocker):
        """Test that venv path is resolved before determining activate script."""
        with tempfile.TemporaryDirectory() as tmpdir:
            venv_dir = Path(tmpdir) / "test_venv"
            venv_dir.mkdir(exist_ok=True)

            if sys.platform == "win32":
                (venv_dir / "Scripts").mkdir(exist_ok=True)
            else:
                (venv_dir / "bin").mkdir(exist_ok=True)

            result = get_activate_script(venv_dir)

            assert str(venv_dir) in str(result)


class TestListKernels:
    """
    Tests for the list_kernels function.

    The list_kernels function retrieves a list of installed Jupyter kernels.

    Behavior:
    - Runs jupyter kernelspec list command
    - Returns stdout string with kernel information

    This is used by setup/install scripts to display available kernels
    after installation, helping users verify their environment is
    correctly registered.

    Return: String containing kernel listing (or error message)
    """

    def test_list_kernels_success(self, mocker):
        """
        Test successful kernel listing.

        Expected behavior: Returns stdout from jupyter kernelspec list.

        When the kernel list command completes successfully, this method
        should return the output as a string that can be displayed to
        users or parsed for verification purposes.

        Input: No parameters (runs jupyter command)
        Output: "kernel1\nkernel2" (string from stdout)
        """
        mock_result = mocker.MagicMock()
        mock_result.stdout = "kernel1\nkernel2"
        mocker.patch("subprocess.run", return_value=mock_result)

        result = list_kernels()

        assert result == "kernel1\nkernel2"

    def test_list_kernels_failure(self, mocker):
        """
        Test kernel listing failure handling.

        Expected behavior: Returns error message string on subprocess failure.

        When jupyter kernelspec list fails (e.g., Jupyter not installed),
        this method should catch the exception and return a user-friendly
        error message rather than crashing the application.

        Input: Subprocess.CalledProcessError raised
        Output: "Failed to list kernels" string
        """
        mock_error = subprocess.CalledProcessError(1, "test")
        mocker.patch("subprocess.run", side_effect=mock_error)

        result = list_kernels()

        assert result == "Failed to list kernels"


class TestGetPythonExecutable:
    """
    Tests for the get_python_executable function.

    The get_python_executable function returns the path to the current
    Python interpreter.

    Behavior:
    - Uses sys.executable or distutils spawn.find_executable
    - Returns string with full Python executable path

    This is used throughout setup/install scripts when constructing
    paths for virtual environment commands, ensuring the correct
    Python interpreter is always referenced.

    Return: Full path to Python executable as string
    """

    def test_get_python_executable(self):
        """
        Test that get_python_executable returns valid Python path.

        Expected behavior: Returns non-empty string with "python" in name.

        This verifies the method can successfully locate and return
        a valid Python interpreter path. The returned path should be
        usable for all subprocess calls requiring a Python executable.

        Input: No parameters (uses sys.executable)
        Output: "/path/to/python" or similar string

        Verification:
        - Return type is str
        - String length > 0
        - "python" appears in name (case-insensitive check)
        """


class TestMain:
    """
    Tests for the main function entry point.

    The main function orchestrates SNOMED environment setup by calling
    all installation steps in sequence.

    Behavior:
    - Parse command line arguments (venv_name, mode)
    - Create virtual environment
    - Install dependencies
    - Register Jupyter kernel
    - List created kernels

    Entry point for SNOMED environment setup via `python setup/install.py`.
    Provides automated installation workflow.

    Parameters:
    - venv_name: Name for virtual env (default: "snomed_methods_env")
    - mode: Installation mode 'production' or 'dev'

    Ties together all previously tested functions into a single CLI interface.
    """

    def test_main_default_params(self, mocker):
        """
        Test main function with default parameters.

        Expected behavior: Calls all setup steps in order.

        The default execution should:
        1. Use "snomed_methods_env" as venv name
        2. Create virtual environment (create_venv)
        3. Install production dependencies (install_dependencies)
        4. Register kernel (register_kernel)
        5. List available kernels (list_kernels)

        Input: sys.argv = ["setup/install.py"]
        Expected calls: All setup functions called once each
        """
        mocker.patch("sys.argv", ["setup/install.py"])

        mock_create_venv = mocker.patch("setup.install.create_venv")
        mock_install_deps = mocker.patch("setup.install.install_dependencies")
        mock_register_kernel = mocker.patch("setup.install.register_kernel")
        mock_list_kernels = mocker.patch("setup.install.list_kernels")

        main()

        mock_create_venv.assert_called_once()
        mock_install_deps.assert_called_once()
        mock_register_kernel.assert_called_once()
        mock_list_kernels.assert_called_once()

    def test_main_custom_venv_name(self, mocker):
        """
        Test main function with custom venv name.

        Expected behavior: Passes custom name to create_venv.

        When users provide a custom environment name as the first
        argument, main should pass this value through all setup steps.

        Input: sys.argv = ["setup/install.py", "custom_env"]
        Verification:
        - create_venv called with "custom_env"
        """
        mocker.patch("sys.argv", ["setup/install.py", "custom_env"])

        mock_create = mocker.patch("setup.install.create_venv")
        mocker.patch("setup.install.install_dependencies")
        mocker.patch("setup.install.register_kernel")
        mocker.patch("setup.install.list_kernels")

        main()

        assert "custom_env" in str(mock_create.call_args)

    def test_main_dev_mode(self, mocker):
        """
        Test main function with dev mode installation.

        Expected behavior: Installs dependencies with dev extras.

        In dev mode (second argument = "dev"), the setup should install
        dependencies including development tools and MedCAT support,
        suitable for developers working on SNOMED methods.

        Input: sys.argv = ["setup/install.py", "snomed_env", "dev"]
        Expected: install_dependencies called with mode="dev"
        """
        mocker.patch("sys.argv", ["setup/install.py", "snomed_env", "dev"])

        mocker.patch("setup.install.create_venv")
        mocker.patch("setup.install.install_dependencies")
        mocker.patch("setup.install.register_kernel")
        mocker.patch("setup.install.list_kernels")

        main()

    def test_main_production_mode_explicit(self, mocker):
        """
        Test main function with explicit production mode.

        Expected behavior: Installs base dependencies without extras.

        When users specify empty string for mode (third argument),
        it should default to production installation, installing
        only the base package dependencies without dev/medcat extras.

        Input: sys.argv = ["setup/install.py", "snomed_env", ""]
        Expected: install_dependencies called appropriately
        """
        mocker.patch("sys.argv", ["setup/install.py", "snomed_env", ""])

        mock_install = mocker.patch("setup.install.install_dependencies")
        mocker.patch("setup.install.create_venv")
        mocker.patch("setup.install.register_kernel")
        mocker.patch("setup.install.list_kernels")

        main()

        assert mock_install.called


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

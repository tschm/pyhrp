"""Tests for the .env file and the paths it points to."""

from pathlib import Path

import pytest


@pytest.fixture
def project_root():
    """Fixture that provides the project root directory.

    Returns:
        Path: The path to the project root directory.

    """
    return Path(__file__).parent.parent.parent


@pytest.fixture
def env_content(project_root: Path):
    """Fixture that provides the content of the .env file as a dictionary.

    Returns:
        dict: A dictionary containing the key-value pairs from the .env file.

    """
    # Get the project root directory
    env_file_path = project_root / ".env"

    from dotenv import dotenv_values

    return dotenv_values(env_file_path)


def test_env_file_exists():
    """Tests that the .env file exists in the project root.

    Verifies:
        The .env file exists in the project root directory.
    """
    # Get the project root directory (assuming tests are in src/tests)
    project_root = Path(__file__).parent.parent.parent
    env_file_path = project_root / ".env"

    assert env_file_path.exists(), ".env file does not exist in project root"


@pytest.mark.parametrize("folder_key", ["MARIMO_FOLDER", "SOURCE_FOLDER", "TESTS_FOLDER"])
def test_folder_exists(env_content, project_root, folder_key):
    """Tests that the folder path specified in the .env file exists.

    Args:
        env_content: Dictionary containing the environment variables from .env file.
        project_root: Path to the project root directory.
        folder_key: The key in the .env file for the folder to check.

    Verifies:
        The folder path exists in the project structure.

    """
    # Get the folder path from the env_content fixture
    folder_path = env_content.get(folder_key)
    assert folder_path is not None, f"{folder_key} not found in .env file"

    # Check if the path exists
    full_path = project_root / folder_path
    assert full_path.exists(), f"{folder_key} path '{folder_path}' does not exist"
    assert full_path.is_dir(), f"{folder_key} path '{folder_path}' is not a directory"

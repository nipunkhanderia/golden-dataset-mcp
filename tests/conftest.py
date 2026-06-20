"""Shared fixtures for golden-dataset-studio-mcp tests."""

import pytest


@pytest.fixture
def dataset_path(tmp_path) -> str:
    """A fresh, empty directory for each test — no shared filesystem state."""
    return str(tmp_path)


@pytest.fixture
def initialised_dataset(dataset_path):
    """A dataset_path with init_dataset already called against it."""
    from golden_dataset_mcp.models import InitDatasetInput
    from golden_dataset_mcp.server import init_dataset

    init_dataset(
        InitDatasetInput(dataset_path=dataset_path, name="test-dataset", description="for tests")
    )
    return dataset_path

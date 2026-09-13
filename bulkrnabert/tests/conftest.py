import pytest

from reimp_shared.testing import use_fake_dataset, write_fake_dataset


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    """The miniature dataset, with `hub` reading from it."""
    root = write_fake_dataset(tmp_path / "dataset")
    use_fake_dataset(monkeypatch, root)
    return root

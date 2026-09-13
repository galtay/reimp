import pytest

from reimp_shared.testing import use_fake_dataset, write_fake_dataset


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    """The miniature dataset, with `hub` reading from it."""
    root = write_fake_dataset(tmp_path / "dataset")
    use_fake_dataset(monkeypatch, root)
    return root


@pytest.fixture
def project_organ() -> dict[str, str]:
    """Organs for the miniature dataset's projects, which are not TCGA's: two share one."""
    return {"TCGA-AAA": "lung", "TCGA-BBB": "lung", "TCGA-CCC": "kidney"}

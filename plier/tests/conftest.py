from pathlib import Path

import pytest

from reimp_shared.genesets import GeneSets, write_gmt
from reimp_shared.testing import use_fake_dataset, write_fake_dataset

# The miniature dataset's protein-coding genes: every third of 48, less the _PAR_Y copy.
FAKE_CODING_GENES = [f"GENE{i}" for i in range(0, 46, 3)]


@pytest.fixture
def fake_dataset(tmp_path, monkeypatch):
    """The miniature dataset, with `hub` reading from it."""
    root = write_fake_dataset(tmp_path / "dataset")
    use_fake_dataset(monkeypatch, root)
    return root


@pytest.fixture
def fake_prior(tmp_path) -> Path:
    """A GMT of overlapping sets over the miniature dataset's protein-coding genes.

    The last three genes are in no set, and one symbol names no gene, so
    the unmapped count is 1.
    """
    genes = FAKE_CODING_GENES
    sets = GeneSets(
        names=["SET_A", "SET_B", "SET_C", "SET_D"],
        descriptions=["synthetic"] * 4,
        members=[genes[0:6], genes[3:9], genes[6:12], genes[8:13] + ["NOT_A_GENE"]],
    )
    return write_gmt(tmp_path / "prior.gmt", sets)

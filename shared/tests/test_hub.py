"""The loaders run for real against the fake dataset; `-m network` runs them
against the published one."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from reimp_shared import hub


def test_axes_line_up_with_values(fake_dataset) -> None:
    samples, genes = hub.load_samples(), hub.load_genes()
    for quantification in hub.QUANTIFICATIONS:
        values = hub.load_values(quantification)
        assert values.shape == (len(samples), len(genes))
    assert samples["sample_index"].tolist() == list(range(len(samples)))
    assert genes["gene_index"].tolist() == list(range(len(genes)))


def test_values_keep_their_stored_dtype(fake_dataset) -> None:
    for quantification in hub.COUNTS:
        assert hub.load_values(quantification).dtype == np.int32
    for quantification in hub.NORMALIZED:
        assert hub.load_values(quantification).dtype == np.float32


def test_unknown_quantification_raises() -> None:
    with pytest.raises(ValueError, match="unknown quantification"):
        hub.load_values("raw_counts")


def test_rows_out_of_sample_index_order_raise(tmp_path, monkeypatch) -> None:
    table = pa.table(
        {
            "sample_index": pa.array([1, 0], pa.int32()),
            "values": pa.array([[1, 2], [3, 4]], pa.list_(pa.int32())),
        }
    )
    (tmp_path / "unstranded").mkdir()
    pq.write_table(table, tmp_path / "unstranded" / "data.parquet")
    monkeypatch.setattr(
        hub, "parquet_path", lambda config, revision=None: tmp_path / config / "data.parquet"
    )
    with pytest.raises(ValueError, match="sample_index order"):
        hub.load_values("unstranded")


@pytest.mark.network
def test_published_dataset_axes_line_up() -> None:
    samples, genes = hub.load_samples(), hub.load_genes()
    assert len(genes) == 60_660  # GENCODE v36
    assert len(samples) > 10_000
    assert hub.load_values("unstranded").shape == (len(samples), len(genes))


@pytest.mark.network
def test_published_par_y_rows_are_all_zero() -> None:
    """`select_genes` drops _PAR_Y rows on the strength of this."""
    genes = hub.load_genes()
    par_y = genes["gene_id"].str.endswith("_PAR_Y").to_numpy()
    assert par_y.sum() > 0
    assert (hub.load_values("unstranded")[:, par_y] == 0).all()

import numpy as np
import pandas as pd
import pytest
import torch

from reimp_shared.preprocess import (
    log_normalize,
    read_gene_ids,
    select_genes,
    transform_values,
)


@pytest.fixture
def genes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gene_index": range(6),
            "gene_id": ["ENSG1.1", "ENSG2.3", "ENSG3.1", "ENSG1.1_PAR_Y", "ENSG4.2", "ENSG5.1"],
            "gene_type": [
                "protein_coding",
                "lncRNA",
                "protein_coding",
                "protein_coding",
                "miRNA",
                "protein_coding",
            ],
        }
    )


# ---------- log_normalize ----------


def test_log_normalize_rows_sum_to_library_size() -> None:
    counts = np.array([[1, 2, 7], [0, 5, 5]])
    out = log_normalize(counts, library_size=100.0)
    np.testing.assert_allclose(np.expm1(out).sum(axis=1), 100.0, rtol=1e-5)


def test_log_normalize_ignores_sequencing_depth() -> None:
    counts = np.array([[1, 2, 7], [0, 5, 5]])
    np.testing.assert_allclose(log_normalize(counts), log_normalize(counts * 3), rtol=1e-6)


def test_log_normalize_numpy_and_torch_agree() -> None:
    counts = np.random.default_rng(0).poisson(20.0, size=(4, 30)).astype(np.int32)
    out_np = log_normalize(counts)
    out_torch = log_normalize(torch.from_numpy(counts))
    assert out_np.dtype == np.float32 and out_torch.dtype == torch.float32
    np.testing.assert_allclose(out_np, out_torch.numpy(), rtol=1e-5)


def test_log_normalize_zero_row_stays_zero() -> None:
    out = log_normalize(np.array([[0, 0, 0], [1, 1, 2]]))
    assert np.isfinite(out).all()
    assert (out[0] == 0).all()


# ---------- transform_values ----------


def test_transform_none_returns_values_as_stored() -> None:
    values = np.array([[1, 2]], dtype=np.int32)
    assert transform_values(values, "none") is values


def test_transform_log1p() -> None:
    values = np.array([[0.0, 1.0, 9.0]], dtype=np.float32)
    out = transform_values(values, "log1p")
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, np.log1p(values))


def test_transform_lognorm_uses_library_size() -> None:
    out = transform_values(np.array([[1, 3]], dtype=np.int32), "lognorm", library_size=8.0)
    np.testing.assert_allclose(out, np.log1p([[2.0, 6.0]]), rtol=1e-6)


def test_transform_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown transform"):
        transform_values(np.zeros((1, 2)), "zscore")


# ---------- select_genes ----------


def test_select_genes_defaults_to_protein_coding_without_par_y(genes) -> None:
    assert select_genes(genes).tolist() == [0, 2, 5]


def test_select_genes_can_keep_par_y(genes) -> None:
    assert select_genes(genes, drop_par_y=False).tolist() == [0, 2, 3, 5]


def test_select_genes_none_keeps_every_biotype(genes) -> None:
    assert select_genes(genes, gene_types=None).tolist() == [0, 1, 2, 4, 5]


def test_select_genes_accepts_a_single_type_string(genes) -> None:
    assert select_genes(genes, gene_types="miRNA").tolist() == [4]


def test_select_genes_empty_selection_raises(genes) -> None:
    with pytest.raises(ValueError, match="no genes left"):
        select_genes(genes, gene_types=["snRNA"])


def test_select_genes_by_id_follows_the_given_order(genes) -> None:
    # Unversioned "ENSG1" matches the X copy, never the _PAR_Y row.
    assert select_genes(genes, gene_ids=["ENSG5.1", "ENSG2.3", "ENSG1"]).tolist() == [5, 1, 0]


def test_select_genes_ids_override_gene_types(genes) -> None:
    assert select_genes(genes, gene_types=["protein_coding"], gene_ids=["ENSG4"]).tolist() == [4]


def test_select_genes_versioned_id_must_match_exactly(genes) -> None:
    with pytest.raises(ValueError, match="not in the dataset"):
        select_genes(genes, gene_ids=["ENSG2.1"])


def test_select_genes_rejects_duplicate_ids(genes) -> None:
    with pytest.raises(ValueError, match="duplicates"):
        select_genes(genes, gene_ids=["ENSG1", "ENSG1"])


def test_read_gene_ids_skips_comments_and_blanks(tmp_path) -> None:
    path = tmp_path / "genes.txt"
    path.write_text("# vocabulary\nENSG1\n\nENSG2.3  # trailing comment\n")
    assert read_gene_ids(str(path)) == ["ENSG1", "ENSG2.3"]

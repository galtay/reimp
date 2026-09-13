import numpy as np
import pytest
import torch

from reimp_shared import hub
from reimp_shared.data import ExpressionDataModule, load_expression
from reimp_shared.preprocess import log_normalize, select_genes
from reimp_shared.splits import N_FOLDS
from reimp_shared.testing import NORMAL, PROJECTS

# ---------- load_expression ----------


def test_defaults_are_raw_protein_coding_counts(fake_dataset) -> None:
    data = load_expression()
    genes = hub.load_genes()
    expected = select_genes(genes)
    assert set(data.genes["gene_type"]) == {"protein_coding"}
    assert data.genes["gene_index"].tolist() == expected.tolist()
    assert data.values.dtype == np.int32
    np.testing.assert_array_equal(data.values, hub.load_values("unstranded")[:, expected])


def test_every_sample_gets_a_split(fake_dataset) -> None:
    data = load_expression()
    assert len(data.samples) == len(hub.load_samples())
    assert set(data.samples["split"]) == {"train", "val", "test"}
    rows = np.concatenate([data.rows(split) for split in ("train", "val", "test")])
    assert sorted(rows.tolist()) == list(range(len(data.samples)))


def test_quantification_and_transform(fake_dataset) -> None:
    columns = select_genes(hub.load_genes())
    tpm = load_expression("tpm_unstranded", transform="log1p")
    assert tpm.values.dtype == np.float32
    expected = np.log1p(hub.load_values("tpm_unstranded")[:, columns])
    np.testing.assert_allclose(tpm.values, expected, rtol=1e-6)

    lognorm = load_expression(transform="lognorm", library_size=1e4)
    np.testing.assert_allclose(np.expm1(lognorm.values).sum(axis=1), 1e4, rtol=1e-4)


def test_gene_ids_choose_and_order_columns(fake_dataset) -> None:
    genes = hub.load_genes()
    picks = [5, 2, 9]
    data = load_expression(gene_ids=genes["gene_id"].iloc[picks].tolist())
    assert data.genes["gene_index"].tolist() == picks
    np.testing.assert_array_equal(data.values, hub.load_values("unstranded")[:, picks])


def test_all_genes(fake_dataset) -> None:
    data = load_expression(gene_types=None, drop_par_y=False)
    assert data.values.shape[1] == len(hub.load_genes())


def test_sample_filters_keep_rows_aligned(fake_dataset) -> None:
    data = load_expression(projects=[PROJECTS[0]], sample_types=[NORMAL], gene_types=None)
    assert set(data.samples["project_id"]) == {PROJECTS[0]}
    assert set(data.samples["sample_type"]) == {NORMAL}
    raw = hub.load_values("unstranded")[data.samples["sample_index"].to_numpy()]
    np.testing.assert_array_equal(data.values, raw[:, data.genes["gene_index"].to_numpy()])


def test_sample_filters_that_match_nothing_raise(fake_dataset) -> None:
    with pytest.raises(ValueError, match="no samples"):
        load_expression(projects=["TCGA-NONE"])


def test_split_does_not_depend_on_what_is_loaded(fake_dataset) -> None:
    """Two models on different quantifications, genes and samples hold out the same patients."""
    a = load_expression().samples
    b = load_expression(
        "tpm_unstranded", gene_types=None, transform="log1p", projects=[PROJECTS[1]]
    ).samples
    merged = b.merge(a, on="sample_index", suffixes=("_b", "_a"))
    assert len(merged) == len(b)
    assert (merged["split_b"] == merged["split_a"]).all()


@pytest.mark.parametrize("fold", range(N_FOLDS))
def test_a_fold_tests_its_own_cases(fold, fake_dataset) -> None:
    data = load_expression(fold=fold)
    assert set(data.samples["split"]) == {"train", "val", "test"}
    np.testing.assert_array_equal(data.rows("test"), np.flatnonzero(data.samples["fold"] == fold))


def test_fold_column_does_not_depend_on_the_fold_loaded(fake_dataset) -> None:
    folds = [load_expression(fold=k).samples["fold"] for k in (0, 3)]
    assert folds[0].equals(folds[1])
    assert set(folds[0]) == set(range(N_FOLDS))


# ---------- ExpressionDataModule ----------


def test_datamodule_knows_n_genes_before_setup(fake_dataset) -> None:
    dm = ExpressionDataModule()
    assert dm.n_genes == len(select_genes(hub.load_genes()))
    assert dm.data is None


def test_datamodule_batches(fake_dataset) -> None:
    dm = ExpressionDataModule(transform="lognorm", batch_size=4)
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    assert batch["values"].shape == (4, dm.n_genes)
    assert batch["values"].dtype == torch.float32
    assert batch["sample_index"].shape == (4,)
    rows = batch["sample_index"].numpy()
    expected = log_normalize(hub.load_values("unstranded")[rows][:, select_genes(hub.load_genes())])
    np.testing.assert_allclose(batch["values"].numpy(), expected, rtol=1e-5)


def test_datamodule_loaders_follow_the_split(fake_dataset) -> None:
    dm = ExpressionDataModule(batch_size=5)
    dm.setup()
    samples = dm.data.samples.set_index("sample_index")
    seen = {}
    for split, loader in [
        ("val", dm.val_dataloader()),
        ("test", dm.test_dataloader()),
        ("all", dm.predict_dataloader()),
    ]:
        seen[split] = torch.cat([b["sample_index"] for b in loader]).tolist()
    for split in ("val", "test"):
        assert set(samples.loc[seen[split], "split"]) == {split}
    assert seen["all"] == samples.index.tolist()
    # Training drops the last partial batch, so it covers all but < batch_size samples.
    train = torch.cat([b["sample_index"] for b in dm.train_dataloader()]).tolist()
    assert set(samples.loc[train, "split"]) == {"train"}
    assert len(dm.data.rows("train")) - len(train) < 5


def test_datamodule_for_a_fold(fake_dataset) -> None:
    dm = ExpressionDataModule(batch_size=4, fold=1)
    dm.setup()
    samples = dm.data.samples.set_index("sample_index")
    test = torch.cat([b["sample_index"] for b in dm.test_dataloader()]).tolist()
    val = torch.cat([b["sample_index"] for b in dm.val_dataloader()]).tolist()
    train = torch.cat([b["sample_index"] for b in dm.train_dataloader()]).tolist()
    assert sorted(test) == samples.index[samples["fold"] == 1].tolist()
    assert (samples.loc[val, "fold"] != 1).all() and (samples.loc[train, "fold"] != 1).all()
    assert set(samples.loc[val, "split"]) == {"val"}


def test_datamodule_rejects_a_bad_fold(fake_dataset) -> None:
    with pytest.raises(ValueError, match="fold must be"):
        ExpressionDataModule(fold=N_FOLDS)


def test_datamodule_reads_gene_ids_from_a_file(fake_dataset, tmp_path) -> None:
    ids = hub.load_genes()["gene_id"].iloc[[3, 1]].tolist()
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(ids))
    dm = ExpressionDataModule(gene_ids_path=str(path))
    dm.setup()
    assert dm.n_genes == 2
    assert dm.data.genes["gene_id"].tolist() == ids


def test_datamodule_rejects_gene_ids_and_a_path(fake_dataset, tmp_path) -> None:
    with pytest.raises(ValueError, match="not both"):
        ExpressionDataModule(gene_ids=["x"], gene_ids_path=str(tmp_path / "x.txt"))

import hashlib

import pytest

from reimp_shared import genesets
from reimp_shared.genesets import (
    MSIGDB,
    Collection,
    GeneSets,
    load_gene_sets,
    msigdb_path,
    read_gmt,
    write_gmt,
)

TOY = GeneSets(["TOY_A", "TOY_B"], ["a", "b"], [["X", "Y"], ["Y", "Z"]])


@pytest.fixture
def served(tmp_path, monkeypatch):
    """One collection, `toy`, pinned and served from a local directory; the cache in tmp_path."""
    file_name = Collection("c9.toy", "", "").file_name
    path = write_gmt(tmp_path / "server" / genesets.MSIGDB_VERSION / file_name, TOY)
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    monkeypatch.setattr(genesets, "MSIGDB", {"toy": Collection("c9.toy", md5, "toy")})
    monkeypatch.setattr(genesets, "MSIGDB_URL", (tmp_path / "server").as_uri())
    monkeypatch.setenv("REIMP_CACHE", str(tmp_path / "cache"))
    return path


def test_gmt_round_trip(tmp_path) -> None:
    sets = GeneSets(["A", "B cells"], ["one", "two"], [["X", "Y"], ["Z"]])
    assert read_gmt(write_gmt(tmp_path / "sets.gmt", sets)) == sets


def test_read_gmt_drops_repeated_members_and_rejects_repeated_names(tmp_path) -> None:
    path = tmp_path / "sets.gmt"
    path.write_text("A\td\tX\tY\tX\n\nB\td\tZ\n")
    assert read_gmt(path).members == [["X", "Y"], ["Z"]]
    path.write_text("A\td\tX\nA\td\tY\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_gmt(path)


def test_a_collection_is_fetched_once_then_read_from_the_cache(served, tmp_path) -> None:
    path = msigdb_path("toy")
    assert path == tmp_path / "cache" / "msigdb" / served.name
    assert read_gmt(path) == TOY
    served.unlink()
    assert msigdb_path("toy") == path


def test_a_download_that_misses_its_pin_is_refused_and_not_cached(served, tmp_path) -> None:
    served.write_text("TOY_A\ta\tX\n")
    with pytest.raises(ValueError, match="md5"):
        msigdb_path("toy")
    assert not any((tmp_path / "cache" / "msigdb").iterdir())


def test_load_gene_sets_reads_collections_and_files_in_order(served, tmp_path) -> None:
    own = write_gmt(tmp_path / "own.gmt", GeneSets(["MINE"], ["m"], [["Q"]]))
    loaded = load_gene_sets([str(own), "toy"])
    assert loaded.names == ["MINE", "TOY_A", "TOY_B"]
    assert loaded.members == [["Q"], ["X", "Y"], ["Y", "Z"]]
    assert load_gene_sets("toy") == TOY


def test_load_gene_sets_rejects_unknown_sources_and_repeated_names(served) -> None:
    with pytest.raises(ValueError, match="neither an MSigDB collection"):
        load_gene_sets(["toy", "no_such_collection"])
    with pytest.raises(ValueError, match="more than one source"):
        load_gene_sets(["toy", "toy"])


def test_licence_restricted_collections_are_not_listed() -> None:
    stems = [collection.stem for collection in MSIGDB.values()]
    assert len(set(stems)) == len(stems)
    assert not any("kegg_legacy" in stem or "biocarta" in stem for stem in stems)


@pytest.mark.network
@pytest.mark.parametrize("name", sorted(MSIGDB))
def test_every_collection_downloads_and_matches_its_pin(name, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("REIMP_CACHE", str(tmp_path))
    assert len(read_gmt(msigdb_path(name))) > 0

"""Gene sets: GMT files, and MSigDB collections fetched by name.

A GMT file holds one gene set per line: name, description and member
symbols, tab-separated. `load_gene_sets` reads a list of sources, each
either an MSigDB collection named in `MSIGDB` or the path of a GMT file:

    load_gene_sets(["reactome", "cell_type", "markers/lm22.gmt"])

MSigDB collections come from release 2026.1, the release the ssGSEA
labels in `reimp_shared.labels` were scored on. Each is downloaded once
from the Broad's release server into `$REIMP_CACHE/msigdb/` (default
`~/.cache/reimp`) and checked against the md5 of the file reimp was built
with, so a re-release is refused rather than silently used. A collection
with no pin yet is kept with a warning that names its md5, to be recorded
in `MSIGDB`.

Licences (https://www.gsea-msigdb.org/gsea/msigdb_license_terms.jsp):
MSigDB is CC BY 4.0, and `kegg_medicus` is CC BY-SA 4.0. C2:CP's
KEGG_LEGACY and BioCarta sub-collections, which KEGG and BioCarta license
to the Broad Institute alone, are deliberately not listed.
"""

from __future__ import annotations

import hashlib
import urllib.request
import warnings
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from reimp_shared.labels import cache_dir

MSIGDB_VERSION = "2026.1.Hs"
MSIGDB_URL = "https://data.broadinstitute.org/gsea-msigdb/msigdb/release"


@dataclass
class GeneSets:
    """Named gene sets, as a GMT file holds them: name, description, members."""

    names: list[str]
    descriptions: list[str]
    members: list[list[str]]

    def __len__(self) -> int:
        return len(self.names)


def read_gmt(path: Path | str) -> GeneSets:
    """Gene sets from a GMT file: one per line, tab-separated `name`, `description`, genes."""
    names, descriptions, members = [], [], []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        name, description, *genes = line.rstrip("\n").split("\t")
        names.append(name)
        descriptions.append(description)
        members.append(list(dict.fromkeys(g for g in genes if g)))
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: duplicate gene-set names")
    return GeneSets(names, descriptions, members)


def write_gmt(path: Path | str, sets: GeneSets) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "\t".join([name, description, *genes])
        for name, description, genes in zip(
            sets.names, sets.descriptions, sets.members, strict=True
        )
    ]
    path.write_text("\n".join(lines) + "\n")
    return path


@dataclass(frozen=True)
class Collection:
    """One MSigDB collection: its file stem, the md5 it is pinned to, what it holds."""

    stem: str
    md5: str | None
    description: str

    @property
    def file_name(self) -> str:
        return f"{self.stem}.v{MSIGDB_VERSION}.symbols.gmt"


MSIGDB: dict[str, Collection] = {
    "hallmark": Collection(
        "h.all", "367eec875967c2cfbf664a1a065b7b8d", "H: 50 hallmark signatures"
    ),
    "reactome": Collection(
        "c2.cp.reactome", "1516b5d15611415d1996c92b7cb6d1cc", "C2:CP:REACTOME: 1,839 pathways"
    ),
    "pid": Collection(
        "c2.cp.pid", "291508046f73d82d13e5efb47492fa47", "C2:CP:PID: 196 NCI-Nature pathways"
    ),
    "wikipathways": Collection(
        "c2.cp.wikipathways", "8e8a38972816a3997a557d6dd625138a", "C2:CP:WIKIPATHWAYS: 925 pathways"
    ),
    # Not pinned yet: the release server answered 503 when these two were added.
    "kegg_medicus": Collection(
        "c2.cp.kegg_medicus", None, "C2:CP:KEGG_MEDICUS: KEGG's openly licensed MEDICUS pathways"
    ),
    "cell_type": Collection("c8.all", None, "C8: cell type signatures from single-cell studies"),
    "oncogenic": Collection(
        "c6.all", "aba0e2214ff63327ae3fb0ce4bcd11c2", "C6: 189 oncogenic signatures"
    ),
    "cancer_cell_atlas": Collection(
        "c4.3ca", "ff9902288655ff2ab88fcb5cbc4a95dd", "C4:3CA: 148 cancer cell atlas meta-programs"
    ),
}


def msigdb_path(name: str) -> Path:
    """The cached GMT file of MSigDB collection `name`, downloaded on first use."""
    if name not in MSIGDB:
        raise ValueError(f"unknown MSigDB collection {name!r}; expected one of {sorted(MSIGDB)}")
    collection = MSIGDB[name]
    path = cache_dir() / "msigdb" / collection.file_name
    if path.exists():
        _check(path, collection, f"delete {path} to download it again")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    url = f"{MSIGDB_URL}/{MSIGDB_VERSION}/{collection.file_name}"
    with urllib.request.urlopen(url, timeout=120) as response:
        partial.write_bytes(response.read())
    try:
        _check(partial, collection, "MSigDB may have re-released it; inspect before re-pinning")
    except ValueError:
        partial.unlink()
        raise
    return partial.replace(path)


def _check(path: Path, collection: Collection, advice: str) -> None:
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    if collection.md5 is None:
        warnings.warn(f"{collection.file_name} is not pinned; its md5 is {digest}", stacklevel=3)
    elif digest != collection.md5:
        raise ValueError(f"{path}: md5 {digest}, pinned {collection.md5}; {advice}")


def load_gene_sets(sources: Sequence[str]) -> GeneSets:
    """Every set of every source, in order: MSigDB collections by name, GMT files by path."""
    if isinstance(sources, str):
        sources = [sources]
    names, descriptions, members = [], [], []
    for source in sources:
        if source in MSIGDB:
            path = msigdb_path(source)
        elif Path(source).is_file():
            path = Path(source)
        else:
            raise ValueError(
                f"{source!r} is neither an MSigDB collection {sorted(MSIGDB)} nor a GMT file"
            )
        sets = read_gmt(path)
        names += sets.names
        descriptions += sets.descriptions
        members += sets.members
    repeated = sorted(name for name, n in Counter(names).items() if n > 1)
    if repeated:
        raise ValueError(f"{len(repeated)} gene sets in more than one source, e.g. {repeated[:5]}")
    return GeneSets(names, descriptions, members)

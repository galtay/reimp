# COMPASS concept hierarchy

`conception_processed.tsv` is COMPASS's gene → gene set → concept table,
copied unchanged from the COMPASS code repository:

- source: <https://github.com/mims-harvard/COMPASS/blob/0e5c87665247e3a300f28282c8bbcc14e26973bd/compass/tokenizer/conception_processed.tsv>
- repository commit `0e5c87665247e3a300f28282c8bbcc14e26973bd` (2026-05-19);
  the file last changed in `7b6d8e5e10a7a539192d2cfd846fbe672fcf0f15`
  (2024-07-18)
- sha256 `a1ae780162ea302600fe095e0e5da15563a49d11ba27266b1e03e1ad607c70b2`
- licence: MIT, Copyright (c) 2023 Artificial Intelligence for Medicine and
  Science @ Harvard. The full notice is in [`LICENSE-COMPASS`](LICENSE-COMPASS),
  as the licence requires. The same sets are the paper's Supplementary
  Data 1; each row's `Reference` column cites the study it was curated from.

What `reimp_compass.hierarchy` reads from it: one row per gene set
(`GeneSet`), its member genes by HGNC symbol (`Genes`, colon-separated), the
concept it feeds (`BroadCelltypePathway`), and the order of both
(`GeneSet_index`, `Concept_index`). `Lineage`, `Description`, `Reference`
and `n_genes` are carried along unread.

- 132 sets, 1–51 genes each (median 7), 1,283 memberships over 916 unique
  genes. All 916 symbols are protein-coding genes in the dataset's GENCODE
  v36 annotation.
- 43 concepts; `Reference` (3 housekeeping sets) is last.
- `GeneSet_index` runs 0–133 with 54 and 55 unused; rows are in its order.
- `SLC4A10` appears twice in `Tcell_IL7Rmax_sc` and `XCR1` twice in
  `cDC1_sc`. Both duplicates are kept, as COMPASS keeps them.

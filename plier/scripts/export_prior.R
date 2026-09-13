# Export the gene-set prior reimp's PLIER uses (plier/paper.md, "For reimp")
# as a GMT file:
#
#   Rscript plier/scripts/export_prior.R [plier/priors/recommended.gmt]
#
# Reads three binary gene x gene-set matrices bundled with the PLIER R
# package (wgmao/PLIER, GPL >= 2) at a pinned commit, straight from GitHub,
# so the package itself need not be installed:
#
#   bloodCellMarkersIRISDMAP   61 cell-type marker sets, from IRIS and DMAP
#   svmMarkers                 22 sets, CIBERSORT LM22
#   canonicalPathways          545 MSigDB C2:CP sets, less the 252 REACTOME_*
#                              and 116 PID_* sets, which reimp's pathway probe
#                              scores: 177 left (122 KEGG, 25 BioCarta, 30 other)
#
# 260 sets in all. Genes are the package's HGNC symbols, sorted; each line's
# description records its source object and the commit. MultiPLIER's prior is
# the same three objects with REACTOME and PID kept.

commit <- "fe4e9b23c47ee199afc5c984692df79ac5aabe80"
md5 <- c(
  bloodCellMarkersIRISDMAP = "28302e66908a223a2e2e2340f9ffdb23",
  svmMarkers = "2daa59404dfd99abbc9503c9a983fce2",
  canonicalPathways = "df4901d15fafeed55d4d800a7b8c0df3"
)
excluded_prefixes <- c("REACTOME", "PID")

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args)) args[1] else "plier/priors/recommended.gmt"

lines <- character()
for (name in names(md5)) {
  path <- file.path(tempdir(), paste0(name, ".rda"))
  url <- sprintf("https://raw.githubusercontent.com/wgmao/PLIER/%s/data/%s.rda", commit, name)
  download.file(url, path, mode = "wb", quiet = TRUE)
  if (unname(tools::md5sum(path)) != md5[[name]]) stop(name, ".rda: md5 mismatch")
  env <- new.env()
  load(path, envir = env)
  mat <- get(name, envir = env)
  sets <- colnames(mat)
  if (name == "canonicalPathways") {
    sets <- sets[!sub("_.*", "", sets) %in% excluded_prefixes]
  }
  source <- sprintf("PLIER::%s@%s", name, substr(commit, 1, 7))
  for (set in sets) {
    genes <- sort(rownames(mat)[mat[, set] > 0], method = "radix")
    lines <- c(lines, paste(c(set, source, genes), collapse = "\t"))
  }
}
if (anyDuplicated(sub("\t.*", "", lines))) stop("duplicate gene-set names")
dir.create(dirname(out), recursive = TRUE, showWarnings = FALSE)
writeLines(lines, out)
message(sprintf("wrote %s: %d gene sets", out, length(lines)))

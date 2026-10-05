# Interaction check (Limitations): refit each final model with an added
# interaction between response length and citation diversity (RL x CD).
#
# Usage (from the repository root):
#   Rscript appendix/interaction.R researchqa                # all judges
#   Rscript appendix/interaction.R sciarena --cores 5        # judges in parallel
#   Rscript appendix/interaction.R sciarena llama            # one judge
#
# SciArena fits start from the final models' estimates when sciarena/mem.R has
# cached them, which makes them much faster.
#
# Outputs (appendix/results/interaction/):
#   <dataset>_<judge>.txt   model summary
#   <dataset>.csv           the RL x CD coefficient of each judge

suppressPackageStartupMessages(library(parallel))

args <- commandArgs(trailingOnly = TRUE)
cores_at <- match("--cores", args)
cores <- if (is.na(cores_at)) 1L else as.integer(args[cores_at + 1])
if (!is.na(cores_at)) args <- args[-c(cores_at, cores_at + 1)]
dataset <- args[1]
stopifnot(dataset %in% c("sciarena", "researchqa"))
mem <- new.env()
sys.source(file.path(dataset, "mem.R"), envir = mem)
judges <- if (length(args) > 1) args[-1] else mem$JUDGES

out_dir <- "appendix/results/interaction"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

fit_with_interaction <- function(judge) {
  formula <- as.formula(paste(deparse1(mem$FINAL[[judge]]), "+ RL:CD"))
  cached <- file.path(dataset, "results", "mem", "fits", paste0(judge, ".rds"))
  theta <- if (file.exists(cached)) getME(readRDS(cached), "theta")
  model <- mem$fit_formula(formula, mem$load_data(judge), theta = theta)
  writeLines(capture.output(print(summary(model))), file.path(out_dir, sprintf("%s_%s.txt", dataset, judge)))
  coefs <- summary(model)$coefficients["RL:CD", ]
  data.frame(judge = judge, estimate = coefs[[1]], std_error = coefs[[2]], z = coefs[[3]], p_value = coefs[[4]])
}

rows <- do.call(rbind, mclapply(judges, fit_with_interaction, mc.cores = cores))
print(rows, row.names = FALSE)

csv <- file.path(out_dir, paste0(dataset, ".csv"))
old <- if (file.exists(csv)) read_csv(csv, show_col_types = FALSE) else rows[0, ]
bind_rows(filter(old, !judge %in% rows$judge), rows) %>%
  arrange(match(judge, mem$JUDGES)) %>%
  write_csv(csv)

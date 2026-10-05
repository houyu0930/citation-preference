# Random-effects model selection (Appendix B).
#
# Backward elimination of random-effect components by AIC, keeping all fixed
# effects:
#   1. Fit the starting structure. While the fit is singular, first make
#      correlated terms uncorrelated (| -> ||), then drop the component with the
#      smallest estimated variance.
#   2. Drop the component whose removal lowers AIC the most, and repeat until
#      no removal lowers AIC. Removals that make the fit singular are skipped.
#
# --start maximal (default) runs the selection from the maximal structure of
# §3.1 (SciArena) / §4.1 (ResearchQA). --start final starts from the final
# model of Appendix B, i.e. checks that no single removal lowers its AIC.
#
# Usage (from the repository root):
#   Rscript appendix/model_selection.R researchqa human
#   Rscript appendix/model_selection.R sciarena human --start final --cores 16
#
# A SciArena step fits one model per remaining component (in parallel with
# --cores), and one fit takes tens of minutes, so run SciArena on a cluster.
#
# Output: appendix/results/model_selection/<dataset>_<judge>_<start>.txt

suppressPackageStartupMessages(library(parallel))

args <- commandArgs(trailingOnly = TRUE)
option <- function(name, default) {
  i <- match(name, args)
  if (is.na(i)) default else args[i + 1]
}
dataset <- args[1]
judge <- args[2]
start <- option("--start", "maximal")
cores <- as.integer(option("--cores", "1"))
stopifnot(dataset %in% c("sciarena", "researchqa"), start %in% c("maximal", "final"))

mem <- new.env()
sys.source(file.path(dataset, "mem.R"), envir = mem)
stopifnot(judge %in% mem$JUDGES)
data <- mem$load_data(judge)

out_dir <- "appendix/results/model_selection"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
out <- file.path(out_dir, sprintf("%s_%s_%s.txt", dataset, judge, start))
log_line <- function(...) {
  line <- sprintf(...)
  message(line)
  cat(line, "\n", file = out, append = TRUE, sep = "")
}

# Random-effects structure as a vector of terms such as "(1 | MA)" or
# "(0 + RL | MAB)"; "||" terms are split into one term per component.
bar_terms <- function(formula) {
  vapply(reformulas::findbars(formula), function(b) paste0("(", deparse1(b), ")"), "")
}
decorrelate <- function(terms) {
  if (length(terms) == 0) return(character(0))
  bar_terms(as.formula(paste("~", paste(sub(" | ", " || ", terms, fixed = TRUE), collapse = " + "))))
}

start_formula <- if (start == "final") mem$FINAL[[judge]] else mem$maximal_formula(judge)
fixed <- deparse1(reformulas::nobars(start_formula))
final_terms <- decorrelate(bar_terms(mem$FINAL[[judge]]))

# Starting values for a reduced model: the larger model's estimates of the
# components they share (lme4 names each one "<group>.<term>"), or NULL.
start_theta <- function(formula, from) {
  if (is.null(from) || !inherits(from, "merMod")) return(NULL)
  cnms <- glFormula(formula, data = data, family = binomial())$reTrms$cnms
  wanted <- unlist(Map(function(group, vars) paste(group, vars, sep = "."), names(cnms), cnms))
  theta <- getME(from, "theta")
  if (all(wanted %in% names(theta))) unname(theta[wanted]) else NULL
}

fit <- function(terms, from = NULL) {
  formula <- as.formula(paste(c(fixed, terms), collapse = " + "))
  started <- Sys.time()
  model <- tryCatch(
    if (length(terms) == 0) {
      glm(formula, data = data, family = binomial())
    } else {
      mem$fit_formula(formula, data, theta = start_theta(formula, from))
    },
    error = function(e) NULL
  )
  list(
    terms = terms, model = model,
    aic = if (is.null(model)) Inf else AIC(model),
    singular = !is.null(model) && inherits(model, "merMod") && isSingular(model),
    minutes = as.numeric(difftime(Sys.time(), started, units = "mins"))
  )
}

# Standard deviation of each random-effect component, named by its term.
component_sd <- function(model) {
  vc <- as.data.frame(VarCorr(model))
  vc <- vc[is.na(vc$var2), ]
  group <- sub("\\.[0-9]+$", "", vc$grp)
  terms <- ifelse(vc$var1 == "(Intercept)", sprintf("(1 | %s)", group), sprintf("(0 + %s | %s)", vc$var1, group))
  setNames(vc$sdcor, terms)
}

cat("", file = out)
log_line("Model selection: %s, %s judge, starting from the %s structure", dataset, judge, start)
log_line("Fixed effects: %s", fixed)
cached <- file.path(dataset, "results", "mem", "fits", paste0(judge, ".rds"))  # from sciarena/mem.R
if (start == "final" && file.exists(cached)) {
  model <- readRDS(cached)
  current <- list(terms = bar_terms(start_formula), model = model, aic = AIC(model),
                  singular = isSingular(model), minutes = 0)
} else {
  current <- fit(bar_terms(start_formula))
}
log_line("Start: AIC %.1f%s (%.1f min)\n  %s", current$aic, if (current$singular) ", singular" else "",
         current$minutes, paste(current$terms, collapse = " + "))

# 1. Reach a non-singular fit.
while (current$singular) {
  uncorrelated <- decorrelate(current$terms)
  if (length(uncorrelated) > length(current$terms)) {
    current <- fit(uncorrelated)
    step <- "made correlated terms uncorrelated"
  } else {
    smallest <- names(which.min(component_sd(current$model)))
    current <- fit(setdiff(current$terms, smallest), from = current$model)
    step <- paste("singular; dropped", smallest)
  }
  log_line("%s: AIC %.1f%s", step, current$aic, if (current$singular) ", still singular" else "")
}

# 2. Backward elimination by AIC.
while (length(current$terms) > 0) {
  candidates <- mclapply(current$terms, function(term) fit(setdiff(current$terms, term), from = current$model),
                         mc.cores = cores)
  log_line("Current AIC %.1f; AIC after removing each term:", current$aic)
  for (i in order(sapply(candidates, `[[`, "aic"))) {
    log_line("  %10.1f  %s%s", candidates[[i]]$aic, current$terms[i], if (candidates[[i]]$singular) " (singular)" else "")
  }
  aics <- sapply(candidates, function(c) if (c$singular) Inf else c$aic)
  best <- which.min(aics)
  if (aics[best] >= current$aic) break
  log_line("Removed %s", current$terms[best])
  current <- candidates[[best]]
}

log_line("\nSelected (AIC %.1f):\n  %s", current$aic, paste(c(fixed, current$terms), collapse = " + "))
log_line("Same random effects as the final model in the paper: %s",
         if (setequal(current$terms, final_terms)) "yes" else "NO")

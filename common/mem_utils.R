# Shared helpers for the mixed-effects analyses. All scripts run from the
# repository root.

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(lme4)
  library(car)
  library(marginaleffects)
})

# Paper names of the predictors. Every continuous predictor is the z-scored
# difference Response A - Response B (§2).
PREDICTORS <- c(
  CD = "Citation Diversity",
  CC = "Citation Count",
  CA = "Citation Age",
  CM = "Citation Mismatch",
  RL = "Response Length",
  RC = "Response Content",
  order = "Presentation Order"
)

zscore <- function(x) as.numeric(scale(x))

# Judges named on the command line (default: all).
parse_judges <- function(all_judges) {
  judges <- commandArgs(trailingOnly = TRUE)
  if (length(judges) == 0) return(all_judges)
  unknown <- setdiff(judges, all_judges)
  if (length(unknown) > 0) stop("Unknown judge(s): ", paste(unknown, collapse = ", "))
  judges
}

# Population-level average marginal effects (Williams, 2012): the change in
# P(Response A is chosen) per 1-SD increase in each predictor, in percentage
# points. For mixed models, random effects are set to zero (re.form = NA).
average_marginal_effects <- function(model, judge) {
  ame <- if (inherits(model, "merMod")) avg_slopes(model, re.form = NA) else avg_slopes(model)
  ame <- ame[order(match(ame$term, names(PREDICTORS))), ]
  data.frame(
    judge = judge,
    predictor = unname(PREDICTORS[ame$term]),
    term = ame$term,
    contrast = ame$contrast,
    ame_pp = 100 * ame$estimate,
    ci_low_pp = 100 * ame$conf.low,
    ci_high_pp = 100 * ame$conf.high,
    p_value = ame$p.value
  )
}

significance <- function(p) {
  ifelse(p < 0.001, "***", ifelse(p < 0.01, "**", ifelse(p < 0.05, "*", "")))
}

format_ame <- function(ame) {
  data.frame(
    Predictor = ame$predictor,
    `Delta Win Rate` = sprintf("%+.2f %s", ame$ame_pp, significance(ame$p_value)),
    `95% CI` = sprintf("[%+.2f, %+.2f]", ame$ci_low_pp, ame$ci_high_pp),
    p = ifelse(ame$p_value < 0.001, "<0.001", sprintf("%.3f", ame$p_value)),
    check.names = FALSE
  )
}

# Write the model summary, variance inflation factors, and AMEs to a text file.
write_report <- function(model, ame, path) {
  vifs <- tryCatch(capture.output(print(vif(model))), error = function(e) conditionMessage(e))
  writeLines(c(
    capture.output(print(summary(model))),
    "", "Variance inflation factors:", vifs,
    "", "Average marginal effects (percentage points):",
    capture.output(print(format_ame(ame), row.names = FALSE))
  ), path)
}

# Merge AMEs into one CSV for all judges, replacing rows of re-fitted judges.
update_ame_csv <- function(new, path, judge_order) {
  old <- if (file.exists(path)) read_csv(path, show_col_types = FALSE) else new[0, ]
  bind_rows(filter(old, !judge %in% new$judge), new) %>%
    arrange(match(judge, judge_order), match(term, names(PREDICTORS))) %>%
    write_csv(path)
}

write_session_info <- function(path) {
  writeLines(capture.output(sessionInfo()), path)
}

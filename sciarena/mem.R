# Mixed-effects analysis of SciArena preferences (§3).
#
# Fits the final mixed-effects logistic regression (Appendix B) to the human
# votes and to each LLM judge's votes, and reports average marginal effects on
# the win rate of Response A (Figure 3; exact values in Table 2). For the human
# model it also compares the random-effect ranking of the 23 response models
# with the SciArena Elo ranking (Figure 2).
#
# Usage (from the repository root):
#   Rscript sciarena/mem.R               # human + all four LLM judges
#   Rscript sciarena/mem.R human qwen    # a subset
#
# Outputs (sciarena/results/mem/):
#   <judge>.txt         model summary, VIFs, and AMEs
#   ame.csv             AMEs of all judges
#   model_ranking.csv   random-effect vs. Elo ranking of the response models
# Fitted models are cached in sciarena/results/mem/fits/; delete them to refit.

source("common/mem_utils.R")

JUDGES <- c("human", "llama", "gemma", "qwen", "skywork")
OUT_DIR <- "sciarena/results/mem"

# ---------------------------------------------------------------------------
# Data. Notation follows §3.1: CD citation diversity, CC citation count, CA
# citation age, RL response length, RC response content; MA / MB are the models
# whose responses are shown as Response A / B and MAB the ordered model pair;
# Qtype, Qsubj, and Qcat are the question type, subject, and main category.
# ---------------------------------------------------------------------------

add_predictors <- function(df) {
  df %>% mutate(
    CD = zscore(citation_entropy_difference),
    CC = zscore(citation_num_citations_difference),
    CA = zscore(citation_years_difference),
    RL = zscore(response_char_length_difference),
    RC = zscore(bert_score_difference),
    Qtype = question_type,
    Qsubj = subject,
    Qcat = main_category
  )
}

load_data <- function(judge) {
  features <- read_csv("sciarena/data/features.csv", show_col_types = FALSE)
  if (judge == "human") {
    return(features %>%
      add_predictors() %>%
      mutate(vote = human_vote_int, MA = model_a, MB = model_b, MAB = model_pair))
  }
  # Each pair is judged in both presentation orders. For `b_first` judgments
  # the predictors and model identities are flipped, so that Response A is
  # always the response shown first and `vote` = 1 means the judge chose it.
  judgments <- read_csv(
    sprintf("sciarena/results/judgments/%s.csv", judge),
    col_types = cols(.default = col_character())
  )
  # Keeps the judgments' row order; drops pairs outside the analysis set.
  df <- inner_join(judgments, features, by = c("item_id" = "qid")) %>% add_predictors()
  a_first <- df$order == "a_first"
  df %>% mutate(
    vote = as.integer(choice == "A"),
    across(c(CD, CC, CA, RL, RC), ~ ifelse(a_first, .x, -.x)),
    MA = ifelse(a_first, model_a, model_b),
    MB = ifelse(a_first, model_b, model_a),
    MAB = paste(MA, MB, sep = "_"),
    order = factor(order, levels = c("a_first", "b_first"))
  )
}

# ---------------------------------------------------------------------------
# Models. The final random-effects structures (Appendix B) were selected from
# the maximal structure (§3.1, footnote 6); see appendix/model_selection.R.
# ---------------------------------------------------------------------------

FINAL <- list(
  human = vote ~ RL + CD + CC + CA + RC +
    (1 | MA) + (1 | MB) + (1 | MB:Qtype) + (1 | MA:Qcat) + (1 | MB:Qcat) +
    (RL + CD + CA + RC || MAB) +
    (0 + RL + CC + CA + RC || Qsubj) +
    (CD + RC || Qtype),
  llama = vote ~ RL + CD + CC + CA + RC + order +
    (1 | MA) + (1 | MB) + (1 | MA:Qtype) + (1 | MB:Qtype) + (1 | MA:Qcat) + (1 | MB:Qcat) +
    (RL + CD || MAB) +
    (CD + CC + RC || Qsubj) +
    (RL + CA || Qtype),
  gemma = vote ~ RL + CD + CC + CA + RC + order +
    (1 | MA) + (1 | MB) + (1 | MA:Qtype) + (1 | MB:Qcat) +
    (RL + CD + CC + CA || MAB) +
    (RL + CD + CC + RC || Qsubj) +
    (RL + CD + CC + RC || Qtype),
  qwen = vote ~ RL + CD + CC + CA + RC + order +
    (1 | MA) + (1 | MB) + (1 | MA:Qtype) + (1 | MB:Qtype) + (1 | MA:Qcat) + (1 | MB:Qcat) +
    (RL + CD + CC + RC || MAB) +
    (RL + CD + CA + RC || Qsubj) +
    (RL + CA + RC || Qtype),
  skywork = vote ~ RL + CD + CC + CA + RC + order +
    (1 | MA) + (1 | MB) + (1 | MA:Qtype) + (1 | MB:Qtype) + (1 | MA:Qcat) + (1 | MB:Qcat) +
    (CC + CA + RC || MAB) +
    (RL + CD + CC + CA + RC || Qsubj) +
    (CD + CA + RC || Qtype)
)

maximal_formula <- function(judge) {
  fixed <- if (judge == "human") "vote ~ RL + CD + CC + CA + RC" else "vote ~ RL + CD + CC + CA + RC + order"
  as.formula(paste(
    fixed,
    "+ (1 | MA) + (1 | MB) + (1 | MA:Qtype) + (1 | MB:Qtype) + (1 | MA:Qcat) + (1 | MB:Qcat)",
    "+ (CD + CC + CA + RL + RC | MAB)",
    "+ (CD + CC + CA + RL + RC | Qsubj)",
    "+ (CD + CC + CA + RL + RC || Qtype)"
  ))
}

# `theta` optionally gives starting values for the random-effect parameters.
fit_formula <- function(formula, data, theta = NULL) {
  glmer(
    formula,
    data = data, family = binomial(), nAGQ = 0,
    start = if (!is.null(theta)) list(theta = theta),
    control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
  )
}

fit_final <- function(judge) {
  path <- file.path(OUT_DIR, "fits", paste0(judge, ".rds"))
  if (file.exists(path)) return(readRDS(path))
  model <- fit_formula(FINAL[[judge]], load_data(judge))
  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  saveRDS(model, path)
  model
}

# Figure 2: rank the response models by their random intercepts. A strong model
# raises P(A wins) as Response A and lowers it as Response B, so its strength is
# u_MA - u_MB.
model_ranking <- function(model) {
  re <- ranef(model)
  as_a <- setNames(re$MA[, "(Intercept)"], rownames(re$MA))
  as_b <- setNames(re$MB[, "(Intercept)"], rownames(re$MB))
  tibble(model = names(as_a), random_effect = as_a - as_b[names(as_a)]) %>%
    inner_join(read_csv("sciarena/data/elo_leaderboard.csv", show_col_types = FALSE), by = "model") %>%
    mutate(random_effect_rank = rank(-random_effect), elo_rank = rank(-elo)) %>%
    arrange(random_effect_rank)
}

main <- function(judges) {
  dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
  for (judge in judges) {
    message("== ", judge)
    model <- fit_final(judge)
    ame <- average_marginal_effects(model, judge)
    report <- file.path(OUT_DIR, paste0(judge, ".txt"))
    write_report(model, ame, report)
    update_ame_csv(ame, file.path(OUT_DIR, "ame.csv"), JUDGES)
    print(format_ame(ame), row.names = FALSE)

    if (judge == "human") {
      ranking <- model_ranking(model)
      write_csv(ranking, file.path(OUT_DIR, "model_ranking.csv"))
      rho <- cor(ranking$random_effect_rank, ranking$elo_rank, method = "spearman")
      line <- sprintf("Spearman correlation, random-effect vs. Elo ranking (%d models): %.2f", nrow(ranking), rho)
      message(line)
      write(c("", line), report, append = TRUE)
    }
  }
  write_session_info(file.path(OUT_DIR, "session_info.txt"))
}

if (sys.nframe() == 0L) main(parse_judges(JUDGES))

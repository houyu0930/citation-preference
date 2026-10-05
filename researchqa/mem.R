# Mixed-effects analysis of ResearchQA preferences (§4).
#
# Fits the final models (Appendix B) and reports average marginal effects on
# the win rate of Response A (Figure 4; exact values in Table 3):
#   - Human votes: logistic regression with a random intercept per question
#     field, on the table where Response A is gpt or gemini at random (§4.1).
#   - LLM judges: each pair is judged in both presentation orders. Model
#     selection dropped the field intercept, so the final model is a
#     fixed-effects logistic regression with a presentation-order predictor.
#
# Usage (from the repository root):
#   Rscript researchqa/mem.R               # human + all four LLM judges
#   Rscript researchqa/mem.R human qwen    # a subset
#
# Outputs (researchqa/results/mem/):
#   <judge>.txt   model summary, VIFs, and AMEs
#   ame.csv       AMEs of all judges

source("common/mem_utils.R")

JUDGES <- c("human", "llama", "gemma", "qwen", "skywork")
OUT_DIR <- "researchqa/results/mem"

# Notation (§2): CD citation diversity, CC citation count, CM citation
# mismatch, RL response length, RC response content.
add_predictors <- function(df) {
  df %>% mutate(
    CD = zscore(citation_diversity_difference),
    CC = zscore(citation_cnt_difference),
    CM = zscore(mismatched_citation_cnt_difference),
    RL = zscore(response_char_length_difference),
    RC = zscore(bert_score_difference)
  )
}

load_data <- function(judge) {
  if (judge == "human") {
    return(read_csv("researchqa/data/features_shuffled.csv", show_col_types = FALSE) %>%
      add_predictors() %>%
      mutate(vote = human_vote_int))
  }
  # features.csv has Response A = gpt. For `gemini_first` judgments the
  # predictors are flipped, so that Response A is always the response shown
  # first and `vote` = 1 means the judge chose it.
  judgments <- read_csv(
    sprintf("researchqa/results/judgments/%s.csv", judge),
    col_types = cols(.default = col_character())
  )
  features <- read_csv("researchqa/data/features.csv", show_col_types = FALSE)
  df <- inner_join(judgments, features, by = c("item_id" = "id")) %>% add_predictors()
  gpt_first <- df$order == "gpt_first"
  df %>% mutate(
    vote = as.integer(choice == "A"),
    across(c(CD, CC, CM, RL, RC), ~ ifelse(gpt_first, .x, -.x)),
    order = factor(order, levels = c("gpt_first", "gemini_first"))
  )
}

FINAL <- list(
  human = vote ~ RL + RC + CC + CD + CM + (1 | field),
  llama = vote ~ RL + RC + CC + CD + CM + order,
  gemma = vote ~ RL + RC + CC + CD + CM + order,
  qwen = vote ~ RL + RC + CC + CD + CM + order,
  skywork = vote ~ RL + RC + CC + CD + CM + order
)

maximal_formula <- function(judge) {
  fixed <- if (judge == "human") "vote ~ RL + RC + CC + CD + CM" else "vote ~ RL + RC + CC + CD + CM + order"
  as.formula(paste(fixed, "+ (1 | field)"))
}

# `theta` optionally gives starting values for the random-effect parameters.
fit_formula <- function(formula, data, theta = NULL) {
  if (is.null(reformulas::findbars(formula))) {
    glm(formula, data = data, family = binomial())
  } else {
    glmer(formula, data = data, family = binomial(), start = if (!is.null(theta)) list(theta = theta))
  }
}

main <- function(judges) {
  dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
  for (judge in judges) {
    message("== ", judge)
    model <- fit_formula(FINAL[[judge]], load_data(judge))
    ame <- average_marginal_effects(model, judge)
    write_report(model, ame, file.path(OUT_DIR, paste0(judge, ".txt")))
    update_ame_csv(ame, file.path(OUT_DIR, "ame.csv"), JUDGES)
    print(format_ame(ame), row.names = FALSE)
  }
  write_session_info(file.path(OUT_DIR, "session_info.txt"))
}

if (sys.nframe() == 0L) main(parse_judges(JUDGES))

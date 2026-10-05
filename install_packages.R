# Install the R packages used by sciarena/mem.R and researchqa/mem.R.
# Tested with R 4.5.1, lme4 1.1-38, marginaleffects 0.31.0, car 3.1-3,
# dplyr 1.1.4, readr 2.1.6.
#
#   Rscript install_packages.R

packages <- c("lme4", "marginaleffects", "car", "dplyr", "readr", "reformulas")
missing <- setdiff(packages, rownames(installed.packages()))
if (length(missing) > 0) install.packages(missing, repos = "https://cloud.r-project.org")

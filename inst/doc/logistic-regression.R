## ----include = FALSE----------------------------------------------------------
library(RcppEnsmallen)

knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

# Apply par() settings from the chunk option of the same name, so the
# boilerplate stays out of the displayed code. knitr opens a fresh device per
# chunk, so nothing needs restoring afterwards.
knitr::knit_hooks$set(parset = function(before, options) {
  if (before) do.call(graphics::par, options$parset)
})

## ----example-path, eval = FALSE-----------------------------------------------
# system.file("examples", "logistic-regression.cpp", package = "RcppEnsmallen")

## ----overflow-----------------------------------------------------------------
eta <- 800
log(1 + exp(eta))   # naive
max(eta, 0) + log1p(exp(-abs(eta)))   # stable

## ----sourcecpp, eval = FALSE--------------------------------------------------
# Rcpp::sourceCpp("logistic-regression.cpp")

## ----data-generation----------------------------------------------------------
set.seed(42)

n <- 5000
beta <- c(-1, 0.7, -0.5, 1.2)
p <- length(beta)

# Design matrix with an intercept column
X <- cbind(1, matrix(rnorm(n * (p - 1)), ncol = p - 1))

# Bernoulli responses with the logistic link
y <- rbinom(n, size = 1, prob = 1 / (1 + exp(-(X %*% beta))))

## ----data-estimation----------------------------------------------------------
coefs_ens <- logistic_reg_lbfgs(X, y)
coefs_glm <- glm.fit(X, y, family = binomial())$coefficients

## ----comparison-table, echo = FALSE-------------------------------------------
compare_coefs <- cbind(beta, coefs_ens, unname(coefs_glm))
colnames(compare_coefs) <- c("Truth", "ensmallen", "glm.fit")
rownames(compare_coefs) <- paste0("Beta", seq_len(nrow(compare_coefs)))
knitr::kable(compare_coefs, longtable = FALSE,
             caption = "Comparison of Estimated Coefficients")

## ----agreement----------------------------------------------------------------
max(abs(as.vector(coefs_ens) - coefs_glm))

## ----ridge--------------------------------------------------------------------
lambdas <- c(0, 1, 10, 100, 1000)
ridge_fits <- sapply(lambdas, function(l) logistic_reg_lbfgs(X, y, lambda = l))

## ----ridge-table, echo = FALSE------------------------------------------------
ridge_all <- cbind(ridge_fits, unname(coefs_glm))
ridge_tbl <- rbind(ridge_all,
                   colSums(ridge_all^2),
                   apply(ridge_all, 2, function(b) max(abs(b - coefs_glm))))
colnames(ridge_tbl) <- c(paste0("lambda = ", lambdas), "glm.fit")
rownames(ridge_tbl) <- c(paste0("Beta", seq_len(nrow(ridge_fits))),
                         "Squared norm", "Max deviation from MLE")
knitr::kable(ridge_tbl, digits = 4, longtable = FALSE,
             caption = "Ridge Shrinkage of the Coefficient Vector")

## ----ridge-plot, parset = list(mar = c(4.1, 4.1, 2.1, 6.4)), fig.width = 6.5, fig.height = 4.2, fig.alt = "Four coefficient paths plotted against lambda on an index axis. Every path moves toward zero as lambda grows from 0 to 1000, with the two positive coefficients falling and the two negative ones rising, and none crossing zero."----
cols <- c("#1b6ca8", "#c0392b", "#2e8b57", "#8e44ad")
matplot(seq_along(lambdas), t(ridge_fits), type = "b", pch = 16, lty = 1,
        col = cols, xaxt = "n", xlab = expression(lambda),
        ylab = "Coefficient", main = "Ridge shrinkage")
axis(1, at = seq_along(lambdas), labels = lambdas)
abline(h = 0, col = "grey70", lty = 3)
legend(par("usr")[2], par("usr")[4], xpd = NA, bty = "n", cex = 0.85,
       legend = paste0("Beta", seq_len(nrow(ridge_fits))),
       col = cols, lty = 1, pch = 16)

## ----glmfit-args--------------------------------------------------------------
names(formals(stats::glm.fit))

## ----ridge-agreement----------------------------------------------------------
max(abs(ridge_fits[, 1] - coefs_glm))

## ----step-size----------------------------------------------------------------
set.seed(2024)
step_sizes <- c(0.001, 0.01, 0.05, 0.25)
adam_fits <- sapply(step_sizes,
                    function(s) logistic_reg_adam(X, y, step_size = s))

## ----step-size-table, echo = FALSE--------------------------------------------
step_all <- cbind(adam_fits, unname(coefs_glm))
step_tbl <- rbind(step_all,
                  apply(step_all, 2, function(b) max(abs(b - coefs_glm))))
colnames(step_tbl) <- c(paste0("step = ", step_sizes), "glm.fit")
rownames(step_tbl) <- c(paste0("Beta", seq_len(nrow(adam_fits))),
                        "Max deviation from MLE")
knitr::kable(step_tbl, digits = 4, longtable = FALSE,
             caption = "Adam Coefficients Across Step Sizes")

## ----step-plot, fig.width = 6.5, fig.height = 4.2, fig.alt = "Maximum deviation from the maximum likelihood estimate plotted against Adam step size on logarithmic axes. The curve falls steeply from 0.001, reaches its lowest point at a step size of 0.05, which is circled, and rises again at 0.25."----
dev_from_mle <- apply(adam_fits, 2, function(b) max(abs(b - coefs_glm)))
plot(step_sizes, dev_from_mle, type = "b", pch = 16, log = "xy",
     col = "#1b6ca8", xlab = "Step size",
     ylab = "Max deviation from the MLE", main = "Adam step size")
best <- which.min(dev_from_mle)
points(step_sizes[best], dev_from_mle[best], pch = 1, cex = 2.2,
       col = "#c0392b")

## ----adam-vs-lbfgs, echo = FALSE----------------------------------------------
set.seed(2024)
adam_tbl <- cbind(beta, logistic_reg_adam(X, y), coefs_ens)
colnames(adam_tbl) <- c("Truth", "Adam", "L-BFGS")
rownames(adam_tbl) <- paste0("Beta", seq_len(nrow(adam_tbl)))
knitr::kable(adam_tbl, digits = 4, longtable = FALSE,
             caption = "Mini-Batch Adam Against Full-Batch L-BFGS")

## ----adam-seed----------------------------------------------------------------
set.seed(2024); first  <- logistic_reg_adam(X, y)
set.seed(2024); second <- logistic_reg_adam(X, y)
identical(first, second)


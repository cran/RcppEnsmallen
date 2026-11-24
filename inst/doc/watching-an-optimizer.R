## ----include = FALSE----------------------------------------------------------
library(RcppEnsmallen)

knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

## ----data-generation----------------------------------------------------------
set.seed(42)

n <- 5000
beta <- c(-1, 0.7, -0.5, 1.2)
p <- length(beta)

# Design matrix with an intercept column
X <- cbind(1, matrix(rnorm(n * (p - 1)), ncol = p - 1))

# Bernoulli responses with the logistic link
y <- rbinom(n, size = 1, prob = 1 / (1 + exp(-(X %*% beta))))

## ----reference-fit------------------------------------------------------------
mle <- as.vector(logistic_reg_lbfgs(X, y))
max(abs(mle - unname(glm.fit(X, y, family = binomial())$coefficients)))

## ----verbose------------------------------------------------------------------
set.seed(2024)
fit5 <- logistic_reg_adam(X, y, max_epochs = 5, verbose = TRUE)
max(abs(as.vector(fit5) - mle))

## ----fifty-epochs-------------------------------------------------------------
set.seed(2024)
fit50 <- logistic_reg_adam(X, y)
max(abs(as.vector(fit50) - mle))

## ----example-path, eval = FALSE-----------------------------------------------
# system.file("examples", "logistic-regression.cpp", package = "RcppEnsmallen")

## ----path-run-----------------------------------------------------------------
path <- logistic_reg_lbfgs_path(X, y)
dim(path)

# The last column is the fit logistic_reg_lbfgs() returned earlier.
max(abs(path[, ncol(path)] - mle))

## ----path-plot, fig.width = 6.5, fig.height = 4.6, fig.alt = "Contour plot of the negative log-likelihood over the second and fourth coefficients, with the other two held at their fitted values. The level sets are closed curves nested around a single interior minimum, marked with a red cross. A blue line runs from an open circle at the origin up and to the right across the level sets, joining filled points whose spacing shrinks as they approach the cross, where the line ends."----
# Numerically stable softplus, the R counterpart of the C++ helper.
softplus <- function(e) pmax(e, 0) + log1p(exp(-abs(e)))

# Beta1 and Beta3 stay at their fitted values, so only Beta2 and Beta4 vary.
base <- X[, 1] * mle[1] + X[, 3] * mle[3]

# Grid limits: centered on the optimum, wide enough to hold the whole path.
r2 <- 1.15 * max(abs(path[2, ] - mle[2]))
r4 <- 1.15 * max(abs(path[4, ] - mle[4]))
g2 <- seq(mle[2] - r2, mle[2] + r2, length.out = 55)
g4 <- seq(mle[4] - r4, mle[4] + r4, length.out = 55)

# The negative log-likelihood at every point of the grid.
Z <- outer(g2, g4, Vectorize(function(b2, b4) {
  eta <- base + X[, 2] * b2 + X[, 4] * b4
  sum(softplus(eta) - y * eta)
}))

contour(g2, g4, Z, nlevels = 14, col = "grey65", labcex = 0.6,
        xlab = expression(beta[2]), ylab = expression(beta[4]),
        main = "L-BFGS on the negative log-likelihood")
# The optimum goes down first so the last few steps, which land on top of it,
# stay visible.
points(mle[2], mle[4], col = "#c0392b", pch = 4, cex = 1.8, lwd = 2)
lines(path[2, ], path[4, ], col = "#1b6ca8", lwd = 2)
points(path[2, ], path[4, ], col = "#1b6ca8", pch = 16, cex = 0.7)
points(path[2, 1], path[4, 1], col = "#1b6ca8", pch = 1, cex = 1.8)
legend("topright", bg = "white", box.col = "grey80", cex = 0.8,
       legend = c("Start", "L-BFGS path", "Optimum"),
       col = c("#1b6ca8", "#1b6ca8", "#c0392b"),
       pch = c(1, 16, 4), lty = c(NA, 1, NA))


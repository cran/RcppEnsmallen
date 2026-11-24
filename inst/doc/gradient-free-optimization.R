## ----include = FALSE----------------------------------------------------------
library(RcppEnsmallen)

knitr::opts_chunk$set(
  collapse = TRUE,
  comment = "#>"
)

# Chunks carrying a parset option have those graphics parameters applied
# before the chunk runs. knitr opens a fresh device per chunk, so nothing
# needs restoring afterwards.
knitr::knit_hooks$set(parset = function(before, options) {
  if (before) do.call(graphics::par, options$parset)
})

## ----data-generation----------------------------------------------------------
n <- 400
beta <- c(2, -1.5, 0.8)
p <- length(beta)

set.seed(512)

# Design matrix with an intercept column
X <- cbind(1, matrix(rnorm(n * (p - 1)), ncol = p - 1))

# Continuous responses with heavy tailed errors
y <- as.vector(X %*% beta + rt(n, df = 2))

## ----estimation---------------------------------------------------------------
obj <- function(b) sum(abs(y - X %*% b))

set.seed(512)
fit <- lad_reg_pso(X, y)

nm <- optim(rep(0, p), obj, method = "Nelder-Mead",
            control = list(maxit = 20000, reltol = 1e-12))

## ----comparison-table, echo = FALSE-------------------------------------------
compare_coefs <- cbind(beta, as.vector(fit), nm$par)
colnames(compare_coefs) <- c("Truth", "LBestPSO", "Nelder-Mead")
rownames(compare_coefs) <- paste0("Beta", seq_len(nrow(compare_coefs)))
knitr::kable(compare_coefs, digits = 6, longtable = FALSE,
             caption = "Comparison of Estimated Coefficients")

## ----agreement----------------------------------------------------------------
max(abs(as.vector(fit) - nm$par))

## ----objective----------------------------------------------------------------
c(LBestPSO = obj(fit), NelderMead = nm$value)

obj(fit) - nm$value

## ----residuals----------------------------------------------------------------
resid <- as.vector(y - X %*% fit)
sort(abs(resid))[1:5]

## ----nonvanishing-------------------------------------------------------------
grad <- function(b) -as.vector(t(X) %*% sign(as.vector(y - X %*% b)))

set.seed(9)
step <- rnorm(p)
step <- step / sqrt(sum(step^2))

c(at_the_fit = sqrt(sum(grad(fit)^2)),
  a_step_away = sqrt(sum(grad(fit + 1e-6 * step)^2)))

## ----kink-plot, fig.width = 6.8, fig.height = 3.4, parset = list(mfrow = c(1, 2), mar = c(4.1, 4.3, 3.1, 1.1)), fig.alt = "Two panels. The left panel plots the absolute and squared error objectives against the second coefficient over a window three units wide, both rescaled to a common range; the two curves are bowl shaped and nearly indistinguishable near the bottom. The right panel magnifies the absolute error objective around its minimum, where it is a chain of straight segments joined at marked breakpoints, with a sharp corner at the minimum rather than a smooth turn."----
hold <- X[, 1] * fit[1] + X[, 3] * fit[3]
l1 <- function(b2) sum(abs(y - hold - X[, 2] * b2))
sse <- function(b2) sum((y - hold - X[, 2] * b2)^2)

# Left: both losses over a wide window, each shifted to its own minimum and
# scaled to a common range so they share an axis.
wide <- seq(fit[2] - 1.5, fit[2] + 1.5, length.out = 400)
rescale <- function(v) (v - min(v)) / diff(range(v))
plot(wide, rescale(vapply(wide, l1, 0)), type = "l", lwd = 2, col = "#1b6ca8",
     ylim = c(0, 1), xlab = expression(beta[2]),
     ylab = "Objective, rescaled", main = "Both losses")
lines(wide, rescale(vapply(wide, sse, 0)), lwd = 2, col = "#c0392b", lty = 2)
legend("top", bty = "n", cex = 0.8, legend = c("Absolute", "Squared"),
       col = c("#1b6ca8", "#c0392b"), lty = c(1, 2), lwd = 2)

# Right: the absolute loss near the minimum. A kink sits at every b2 that
# sends one residual through zero.
kinks <- (y - hold) / X[, 2]
halfw <- sort(abs(kinks - fit[2]))[8]
near <- kinks[abs(kinks - fit[2]) <= halfw]
zoom <- seq(fit[2] - halfw, fit[2] + halfw, length.out = 1500)
plot(zoom, vapply(zoom, l1, 0), type = "n", xlab = expression(beta[2]),
     ylab = "Sum of absolute residuals", main = "Absolute, magnified")
abline(v = near, col = "grey80", lty = 3)
lines(zoom, vapply(zoom, l1, 0), lwd = 2, col = "#1b6ca8")
points(near, vapply(near, l1, 0), pch = 16, cex = 0.7, col = "#1b6ca8")
points(fit[2], l1(fit[2]), pch = 4, cex = 1.7, lwd = 2, col = "#c0392b")

## ----quantreg-fit, eval = requireNamespace("quantreg", quietly = TRUE)--------
rq_beta <- as.vector(coef(quantreg::rq(y ~ X - 1, tau = 0.5)))

c(LBestPSO = obj(fit), rq = obj(rq_beta))

obj(fit) - obj(rq_beta)

max(abs(as.vector(fit) - rq_beta))

## ----quantreg-timing, eval = requireNamespace("quantreg", quietly = TRUE)-----
pso_reps <- 10
rq_reps <- 200

pso_time <- system.time(
  for (i in seq_len(pso_reps)) {
    set.seed(512)
    lad_reg_pso(X, y)
  }
)[["elapsed"]] / pso_reps

rq_time <- system.time(
  for (i in seq_len(rq_reps)) quantreg::rq(y ~ X - 1, tau = 0.5)
)[["elapsed"]] / rq_reps

c(LBestPSO = pso_time, rq = rq_time)

pso_time / rq_time

## ----seeding------------------------------------------------------------------
set.seed(20260818); first  <- lad_reg_pso(X, y)
set.seed(20260818); second <- lad_reg_pso(X, y)
identical(first, second)


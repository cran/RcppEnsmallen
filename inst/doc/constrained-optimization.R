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
n <- 100
beta <- c(2, 0, 1.5, 0)
p <- length(beta)

set.seed(24)

# No intercept column: every column of X is constrained
X <- matrix(rnorm(n * p), ncol = p)

# Continuous responses
y <- as.vector(X %*% beta + rnorm(n))

## ----ols----------------------------------------------------------------------
ols <- unname(qr.solve(X, y))
ols

## ----estimation---------------------------------------------------------------
rss <- function(b) sum((y - X %*% b)^2)

fit <- nnls_auglag(X, y)

box <- optim(rep(0, p), rss, method = "L-BFGS-B", lower = rep(0, p),
             control = list(factr = 1e-12, pgtol = 0, maxit = 10000))

## ----comparison-table, echo = FALSE-------------------------------------------
compare_coefs <- cbind(beta, ols, as.vector(fit), box$par)
colnames(compare_coefs) <- c("Truth", "Least squares", "ensmallen", "optim")
rownames(compare_coefs) <- paste0("Beta", seq_len(nrow(compare_coefs)))
knitr::kable(compare_coefs, digits = 6, longtable = FALSE,
             caption = "Comparison of Estimated Coefficients")

## ----agreement----------------------------------------------------------------
max(abs(as.vector(fit) - box$par))

rss(fit) - box$value

## ----objective----------------------------------------------------------------
c(unconstrained = rss(ols), constrained = rss(fit))

## ----coef-plot, fig.width = 6.5, fig.height = 4.2, parset = list(mar = c(4.1, 4.1, 3.1, 1.1)), fig.alt = "Estimated coefficients plotted against coefficient index. The region below zero is shaded and labelled infeasible. For the first and third coefficients the least squares and constrained estimates sit close together well above zero. For the second and fourth the least squares estimates lie inside the shaded region, and arrows carry them up to constrained estimates resting on the zero line, where the true values also lie."----
idx <- seq_len(p)
ylim <- range(c(ols, as.vector(fit), beta)) + c(-0.15, 0.15)
plot(idx, ols, type = "n", xlim = c(0.7, p + 0.3), ylim = ylim, xaxt = "n",
     xlab = "Coefficient", ylab = "Estimate",
     main = "Unconstrained and constrained fits")
# Shade the half of the plane the constraints rule out.
rect(par("usr")[1], par("usr")[3], par("usr")[2], 0, col = "grey93",
     border = NA)
box()
abline(h = 0, col = "grey40")
axis(1, at = idx, labels = paste0("Beta", idx))
arrows(idx, ols, idx, as.vector(fit), length = 0.08, col = "grey55",
       lwd = 1.5)
points(idx, ols, pch = 1, cex = 1.6, lwd = 2, col = "#c0392b")
points(idx, as.vector(fit), pch = 16, cex = 1.4, col = "#1b6ca8")
points(idx, beta, pch = 3, cex = 1.2, col = "grey30")
legend("topright", bty = "n", cex = 0.85,
       legend = c("Least squares", "Non-negative", "Truth"),
       col = c("#c0392b", "#1b6ca8", "grey30"), pch = c(1, 16, 3),
       pt.cex = c(1.6, 1.4, 1.2))
text(par("usr")[1] + 0.1, ylim[1] + 0.06, "infeasible", adj = 0, cex = 0.8,
     col = "grey40")

## ----boundary-----------------------------------------------------------------
as.vector(fit)

min(as.vector(fit))

## ----feasible-----------------------------------------------------------------
set.seed(101)
X_feas <- matrix(rnorm(300 * 3), ncol = 3)
y_feas <- as.vector(X_feas %*% c(3, 2, 5) + rnorm(300, sd = 0.5))

ols_feas <- unname(qr.solve(X_feas, y_feas))
min(ols_feas)

max(abs(as.vector(nnls_auglag(X_feas, y_feas)) - ols_feas))

## ----nnls-fit, eval = requireNamespace("nnls", quietly = TRUE)----------------
lawson_hanson <- nnls::nnls(X, y)

c(ensmallen = rss(fit), nnls = rss(lawson_hanson$x))

rss(fit) - rss(lawson_hanson$x)

max(abs(as.vector(fit) - lawson_hanson$x))

## ----nnls-zeros, eval = requireNamespace("nnls", quietly = TRUE)--------------
lawson_hanson$x

as.vector(fit)

## ----nnls-timing, eval = requireNamespace("nnls", quietly = TRUE)-------------
ens_reps <- 200
nnls_reps <- 2000

t_ens <- system.time(
  for (i in seq_len(ens_reps)) nnls_auglag(X, y)
)[["elapsed"]] / ens_reps

t_nnls <- system.time(
  for (i in seq_len(nnls_reps)) nnls::nnls(X, y)
)[["elapsed"]] / nnls_reps

c(ensmallen = t_ens, nnls = t_nnls)


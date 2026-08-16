#!/usr/bin/env Rscript
# Generates reference values from R (meta 8.x / metafor 4.x) that the Python stats engine must reproduce.
# Output: validation/fixtures/r_reference.json  (consumed by tests/test_stats_vs_r.py)
suppressPackageStartupMessages({ library(meta); library(metafor); library(jsonlite) })

here <- function(p) file.path(dirname(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE)[1])), p)
ref_dir <- here("../reference/cisneros2024")
out <- list(meta_version = as.character(packageVersion("meta")),
            metafor_version = as.character(packageVersion("metafor")))

## ---------- 1. Random-effects pooling on the Cisneros gold tables ----------
pool_fixture <- function(csv, label) {
  tab <- read.csv(csv, check.names = FALSE, stringsAsFactors = FALSE)
  res <- list(k = nrow(tab), TE = tab$TE, seTE = tab$seTE, study = tab$Author)
  for (m in c("REML", "DL", "PM")) {
    for (hk in c(FALSE, TRUE)) {
      mg <- metagen(TE = TE, seTE = seTE, studlab = Author, data = tab, sm = "SMD",
                    common = FALSE, random = TRUE, method.tau = m,
                    method.random.ci = if (hk) "HK" else "classic",
                    prediction = TRUE, method.predict = "HTS")
      key <- paste0(m, if (hk) "_HK" else "")
      res[[key]] <- list(
        TE.random = mg$TE.random, seTE.random = mg$seTE.random,
        lower.random = mg$lower.random, upper.random = mg$upper.random,
        statistic.random = mg$statistic.random, pval.random = mg$pval.random,
        tau2 = mg$tau2, se.tau2 = mg$se.tau2, tau = mg$tau,
        lower.tau2 = mg$lower.tau2, upper.tau2 = mg$upper.tau2,
        I2 = mg$I2, lower.I2 = mg$lower.I2, upper.I2 = mg$upper.I2,
        H = mg$H, Q = mg$Q, df.Q = mg$df.Q, pval.Q = mg$pval.Q,
        lower.predict = mg$lower.predict, upper.predict = mg$upper.predict,
        df.predict = mg$df.predict,
        w.random = as.numeric(mg$w.random),
        w.random.pct = as.numeric(100 * mg$w.random / sum(mg$w.random)),
        TE.common = mg$TE.common, seTE.common = mg$seTE.common,
        lower.common = mg$lower.common, upper.common = mg$upper.common
      )
    }
  }
  # metafor cross-check (REML) + predict()
  rm <- rma(yi = TE, vi = seTE^2, data = tab, method = "REML")
  pr <- predict(rm)
  res$metafor_REML <- list(b = as.numeric(rm$b), se = rm$se, ci.lb = rm$ci.lb, ci.ub = rm$ci.ub,
                           zval = rm$zval, pval = rm$pval, tau2 = rm$tau2, se.tau2 = rm$se.tau2,
                           I2 = rm$I2, H2 = rm$H2, QE = rm$QE, QEp = rm$QEp,
                           pi.lb = pr$pi.lb, pi.ub = pr$pi.ub,
                           weights_pct = as.numeric(weights(rm)))
  rmk <- rma(yi = TE, vi = seTE^2, data = tab, method = "REML", test = "knha")
  res$metafor_REML_knha <- list(b = as.numeric(rmk$b), se = rmk$se, ci.lb = rmk$ci.lb, ci.ub = rmk$ci.ub,
                                zval = rmk$zval, pval = rmk$pval)
  # Egger regression as done in the Cisneros Rmd
  eg <- summary(lm(I(TE / seTE) ~ I(1 / seTE), data = tab))
  res$egger_rmd <- list(intercept = eg$coefficients[1, 1], intercept_se = eg$coefficients[1, 2],
                        intercept_t = eg$coefficients[1, 3], intercept_p = eg$coefficients[1, 4],
                        slope = eg$coefficients[2, 1])
  # metabias (Egger) via meta
  mgr <- metagen(TE = TE, seTE = seTE, data = tab, sm = "SMD", common = FALSE, random = TRUE, method.tau = "REML")
  mb <- metabias(mgr, method.bias = "Egger", k.min = 3)
  res$metabias_egger <- list(statistic = as.numeric(mb$statistic), pval = mb$p.value,
                             intercept = as.numeric(mb$estimate[1]))
  res
}
out$cisneros_late <- pool_fixture(file.path(ref_dir, "late_gsheet.csv"), "late")
out$cisneros_aft  <- pool_fixture(file.path(ref_dir, "aft_gsheet.csv"), "aft")

## ---------- 2. Small synthetic k=5 example (hand-checkable) ----------
yi <- c(-0.5, 0.2, 0.8, -0.1, 0.4); vi <- c(0.09, 0.04, 0.16, 0.0625, 0.0225)
syn <- list(yi = yi, vi = vi)
for (m in c("REML", "DL", "PM", "FE")) {
  r <- rma(yi = yi, vi = vi, method = m)
  pr <- predict(r)
  syn[[m]] <- list(b = as.numeric(r$b), se = r$se, ci.lb = r$ci.lb, ci.ub = r$ci.ub, zval = r$zval, pval = r$pval,
                   tau2 = r$tau2, I2 = r$I2, H2 = r$H2, QE = r$QE, QEp = r$QEp,
                   pi.lb = pr$pi.lb, pi.ub = pr$pi.ub, weights_pct = as.numeric(weights(r)))
}
mg <- metagen(TE = yi, seTE = sqrt(vi), sm = "SMD", common = TRUE, random = TRUE, method.tau = "REML",
              prediction = TRUE, method.predict = "HTS")
syn$meta_REML <- list(TE.random = mg$TE.random, seTE.random = mg$seTE.random, lower.random = mg$lower.random,
                      upper.random = mg$upper.random, tau2 = mg$tau2, I2 = mg$I2, Q = mg$Q,
                      lower.predict = mg$lower.predict, upper.predict = mg$upper.predict, df.predict = mg$df.predict,
                      TE.common = mg$TE.common, seTE.common = mg$seTE.common)
out$synthetic_k5 <- syn

## ---------- 3. SMD from means/SDs: escalc + metacont (Cohen vs Hedges) ----------
smd_cases <- list(
  list(m1 = 31.51, sd1 = 11.12, n1 = 12, m2 = 12.28, sd2 = 11.82, n2 = 12),  # Bock-like
  list(m1 = 8.806, sd1 = 9.34, n1 = 108, m2 = 5.597, sd2 = 7.79, n2 = 109),  # large n
  list(m1 = 2.991, sd1 = 0.948, n1 = 5, m2 = 1.886, sd2 = 1.876, n2 = 5),    # tiny n
  list(m1 = 20.0, sd1 = 5.0, n1 = 20, m2 = 20.0, sd2 = 5.0, n2 = 20),        # zero effect
  list(m1 = 46.14, sd1 = 6.08, n1 = 20, m2 = 44.67, sd2 = 8.76, n2 = 19)     # unequal
)
out$smd_cases <- lapply(smd_cases, function(cs) {
  es_g <- escalc(measure = "SMD", m1i = cs$m1, sd1i = cs$sd1, n1i = cs$n1, m2i = cs$m2, sd2i = cs$sd2, n2i = cs$n2)
  es_d <- escalc(measure = "SMD", m1i = cs$m1, sd1i = cs$sd1, n1i = cs$n1, m2i = cs$m2, sd2i = cs$sd2, n2i = cs$n2, correct = FALSE)
  mc_c <- metacont(n.e = cs$n1, mean.e = cs$m1, sd.e = cs$sd1, n.c = cs$n2, mean.c = cs$m2, sd.c = cs$sd2,
                   sm = "SMD", method.smd = "Cohen")
  mc_h <- metacont(n.e = cs$n1, mean.e = cs$m1, sd.e = cs$sd1, n.c = cs$n2, mean.c = cs$m2, sd.c = cs$sd2,
                   sm = "SMD", method.smd = "Hedges")
  mc_h_exact <- metacont(n.e = cs$n1, mean.e = cs$m1, sd.e = cs$sd1, n.c = cs$n2, mean.c = cs$m2, sd.c = cs$sd2,
                   sm = "SMD", method.smd = "Hedges", exact.smd = TRUE)
  c(cs, list(escalc_g_yi = as.numeric(es_g$yi), escalc_g_vi = as.numeric(es_g$vi),
             escalc_d_yi = as.numeric(es_d$yi), escalc_d_vi = as.numeric(es_d$vi),
             metacont_cohen_TE = mc_c$TE, metacont_cohen_seTE = mc_c$seTE,
             metacont_hedges_TE = mc_h$TE, metacont_hedges_seTE = mc_h$seTE,
             metacont_hedges_exact_TE = mc_h_exact$TE, metacont_hedges_exact_seTE = mc_h_exact$seTE))
})

## ---------- 4. Conversions from test statistics (metafor conventions where available) ----------
# d from t (independent samples): d = t*sqrt(1/n1+1/n2)  (metafor: escalc(measure="SMD", ti=, n1i=, n2i=) gives g)
tcases <- list(list(t = 2.5, n1 = 15, n2 = 14), list(t = -1.2, n1 = 30, n2 = 25), list(t = 3.377, n1 = 10, n2 = 10))
out$t_cases <- lapply(tcases, function(cs) {
  es <- escalc(measure = "SMD", ti = cs$t, n1i = cs$n1, n2i = cs$n2)
  c(cs, list(escalc_g_yi = as.numeric(es$yi), escalc_g_vi = as.numeric(es$vi),
             d_raw = cs$t * sqrt(1 / cs$n1 + 1 / cs$n2)))
})
# d from p (two-sided) & df: metafor escalc(measure="SMD", pi=, n1i=, n2i=) (assumes t from p)
pcases <- list(list(p = 0.03, n1 = 20, n2 = 20), list(p = 0.2, n1 = 12, n2 = 12))
out$p_cases <- lapply(pcases, function(cs) {
  es <- escalc(measure = "SMD", pi = cs$p, n1i = cs$n1, n2i = cs$n2)
  c(cs, list(escalc_g_yi = as.numeric(es$yi), escalc_g_vi = as.numeric(es$vi),
             t_from_p = qt(cs$p / 2, df = cs$n1 + cs$n2 - 2, lower.tail = FALSE)))
})

## ---------- 5. Q-profile CI for tau2 and I2 CI on synthetic ----------
r <- rma(yi = yi, vi = vi, method = "REML"); ci <- confint(r)
out$synthetic_k5$tau2_ci <- list(tau2 = ci$random[1, 1], lb = ci$random[1, 2], ub = ci$random[1, 3],
                                 I2 = ci$random[3, 1], I2_lb = ci$random[3, 2], I2_ub = ci$random[3, 3])


## ---------- 6. Amendment C: combining arms, five-number summaries, leave-one-out, modified Egger ----------
# (a) combine_groups (Cochrane 6.5.2.10): two subgroups' summaries must reproduce the whole sample.
set.seed(7)
g1 <- rnorm(12, 31.5, 11.1); g2 <- rnorm(9, 12.3, 11.8); both <- c(g1, g2)
out$combine_raw <- list(m1 = mean(g1), sd1 = sd(g1), n1 = length(g1),
                        m2 = mean(g2), sd2 = sd(g2), n2 = length(g2),
                        mean = mean(both), sd = sd(both), n = length(both))
# escalc on the combined arm vs a control, so the combined values are checked where they are used
ctrl <- list(m = 20.0, sd = 9.0, n = 15)
es_comb <- escalc(measure = "SMD", m1i = mean(both), sd1i = sd(both), n1i = length(both),
                  m2i = ctrl$m, sd2i = ctrl$sd, n2i = ctrl$n, correct = FALSE)
out$combine_raw$control <- ctrl
out$combine_raw$escalc_d_yi <- as.numeric(es_comb$yi)
out$combine_raw$escalc_d_vi <- as.numeric(es_comb$vi)

# (b) five-number summary -> mean/SD: a large normal sample whose true mean/SD are known
set.seed(11)
x <- rnorm(200, 50, 10); qs <- quantile(x, c(0.25, 0.5, 0.75), type = 7)
out$five_number <- list(n = length(x), min = min(x), q1 = qs[[1]], median = qs[[2]],
                        q3 = qs[[3]], max = max(x), mean = mean(x), sd = sd(x))
set.seed(12)
xs <- rnorm(15, 8, 2); qss <- quantile(xs, c(0.25, 0.5, 0.75), type = 7)
out$five_number_small <- list(n = length(xs), min = min(xs), q1 = qss[[1]], median = qss[[2]],
                              q3 = qss[[3]], max = max(xs), mean = mean(xs), sd = sd(xs))

# (c) leave-one-out on the Cisneros late table (metafor::leave1out, REML)
late <- read.csv(file.path(ref_dir, "late_gsheet.csv"), check.names = FALSE, stringsAsFactors = FALSE)
l1o <- leave1out(rma(yi = TE, vi = seTE^2, data = late, method = "REML"))
out$leave_one_out <- list(estimate = as.numeric(l1o$estimate), se = as.numeric(l1o$se),
                          ci.lb = as.numeric(l1o$ci.lb), ci.ub = as.numeric(l1o$ci.ub),
                          tau2 = as.numeric(l1o$tau2), I2 = as.numeric(l1o$I2),
                          Q = as.numeric(l1o$QE), k = nrow(late))

# (d) Egger with the Pustejovsky-Rodgers predictor sqrt((nA+nB)/(nA nB)) = sqrt(1/ni_eff)
nA <- late$N_old; nB <- late$N_young; ni_eff <- nA * nB / (nA + nB)
rt_pr <- regtest(x = late$TE, vi = late$seTE^2, ni = ni_eff, predictor = "sqrtninv", model = "lm")
out$egger_pr <- list(zval = rt_pr$zval, pval = rt_pr$pval, dfs = rt_pr$dfs,
                     est = as.numeric(rt_pr$est), ci.lb = as.numeric(rt_pr$ci.lb),
                     ci.ub = as.numeric(rt_pr$ci.ub),
                     n_a = nA, n_b = nB, TE = late$TE, seTE = late$seTE)
rt_sei <- regtest(x = late$TE, vi = late$seTE^2, predictor = "sei", model = "lm")
out$egger_sei_lm <- list(zval = rt_sei$zval, pval = rt_sei$pval, dfs = rt_sei$dfs)

write_json(out, here("r_reference.json"), auto_unbox = TRUE, digits = 12, pretty = TRUE, na = "null")
cat("wrote", here("r_reference.json"), "\n")

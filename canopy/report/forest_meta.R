#!/usr/bin/env Rscript
#
# Canopy — draw one forest plot with meta::forest.meta, and report what was pooled.
#
# Called two ways by `canopy/report/forest_r.py`:
#
#   Rscript forest_meta.R --probe
#   Rscript forest_meta.R <rows.csv> <options.json> <out_stem>
#
# Both print ONE JSON object on stdout and nothing else; every message, warning and error goes
# to stderr, so the caller can parse stdout without stripping anything. A failure prints
# {"ok":false,"error":...} and exits 1 rather than dying silently: the Python side then draws the
# plot itself and records why, which is a finding, not a crash.
#
# Nothing here decides anything about the review. Every label, column, axis limit and statistical
# convention arrives in options.json, which the protocol wrote; this file turns them into a call.

.canopy <- new.env(parent = emptyenv())
.canopy$res <- 300
.canopy$pointsize <- 10

quote_json <- function(x) {
  x <- gsub("\\", "\\\\", as.character(x), fixed = TRUE)
  x <- gsub("\"", "\\\"", x, fixed = TRUE)
  x <- gsub("\n", "\\n", x, fixed = TRUE)
  x <- gsub("\r", "\\r", x, fixed = TRUE)
  x <- gsub("\t", "\\t", x, fixed = TRUE)
  paste0("\"", x, "\"")
}

# jsonlite writes the result; the probe below must work without it, so it is hand-rolled.
emit <- function(x) {
  cat(jsonlite::toJSON(x, auto_unbox = TRUE, null = "null", na = "null", digits = NA), "\n",
      sep = "")
}

die <- function(message) {
  cat("{\"ok\":false,\"error\":", quote_json(message), "}\n", sep = "")
  quit(save = "no", status = 1)
}

# ------------------------------------------------------------------------------- devices
# R's own svg() and png(type="cairo") are cairo devices, and a machine without the X11 libraries
# has neither even though capabilities("cairo") says TRUE. So each format is opened on a scratch
# file first and the ones that cannot open are reported back as unavailable rather than left as
# an empty file the report would link to.
canopy_pdf <- function(file, width, height) {
  grDevices::pdf(file = file, width = width, height = height, pointsize = .canopy$pointsize)
}

canopy_png <- function(file, width, height) {
  if (requireNamespace("ragg", quietly = TRUE))
    return(ragg::agg_png(filename = file, width = width, height = height, units = "in",
                         res = .canopy$res, pointsize = .canopy$pointsize))
  type <- if (capabilities("aqua")) "quartz" else if (capabilities("cairo")) "cairo" else NULL
  if (is.null(type))
    grDevices::png(filename = file, width = width, height = height, units = "in",
                   res = .canopy$res, pointsize = .canopy$pointsize)
  else
    grDevices::png(filename = file, width = width, height = height, units = "in",
                   res = .canopy$res, pointsize = .canopy$pointsize, type = type)
}

canopy_svg <- function(file, width, height) {
  if (requireNamespace("svglite", quietly = TRUE))
    return(svglite::svglite(filename = file, width = width, height = height,
                            pointsize = .canopy$pointsize))
  grDevices::svg(filename = file, width = width, height = height, pointsize = .canopy$pointsize)
}

DEVICES <- list(pdf = "canopy_pdf", png = "canopy_png", svg = "canopy_svg")

device_error <- function(format, width, height) {
  if (is.null(DEVICES[[format]])) return(sprintf("no device for format '%s'", format))
  scratch <- tempfile(fileext = paste0(".", format))
  problem <- tryCatch({
    withCallingHandlers(
      do.call(DEVICES[[format]], list(file = scratch, width = width, height = height)),
      warning = function(w) stop(conditionMessage(w), call. = FALSE))
    grDevices::dev.off()
    ""
  }, error = function(e) conditionMessage(e))
  while (grDevices::dev.cur() > 1) grDevices::dev.off()
  unlink(scratch)
  problem
}

args <- commandArgs(trailingOnly = TRUE)

# --------------------------------------------------------------------------------- probe
# What the caller needs before it can trust this script: an R, a `meta` new enough for the
# arguments used below, and a jsonlite to carry the options in and the numbers out.
if (length(args) >= 1 && identical(args[1], "--probe")) {
  has_meta <- requireNamespace("meta", quietly = TRUE)
  has_json <- requireNamespace("jsonlite", quietly = TRUE)
  meta_version <- if (has_meta) as.character(utils::packageVersion("meta")) else ""
  new_enough <- has_meta && utils::packageVersion("meta") >= "6.0"
  # which output formats this machine can actually write, so the caller can decide BEFORE
  # spending a render whether R can produce the whole set the report links
  can <- Filter(function(f) !nzchar(device_error(f, 4, 3)), names(DEVICES))
  cat(sprintf(
    "{\"ok\":%s,\"r_version\":%s,\"meta_version\":%s,\"jsonlite\":%s,\"formats\":[%s]}\n",
    if (new_enough && has_json) "true" else "false",
    quote_json(paste(R.version$major, R.version$minor, sep = ".")),
    quote_json(meta_version),
    if (has_json) "true" else "false",
    paste(vapply(can, quote_json, ""), collapse = ",")))
  quit(save = "no", status = 0)
}

if (length(args) != 3)
  die("usage: forest_meta.R <rows.csv> <options.json> <out_stem>   (or --probe)")

if (!requireNamespace("jsonlite", quietly = TRUE)) die("R package 'jsonlite' is not installed")
if (!requireNamespace("meta", quietly = TRUE)) die("R package 'meta' is not installed")
suppressPackageStartupMessages(library(meta))
# the portable contract: every argument used below exists from meta 6.0 on. The version actually
# used is reported back and printed in the report's methods section.
stopifnot(packageVersion("meta") >= "6.0")

csv_path <- args[1]
opt_path <- args[2]
out_stem <- args[3]

opt <- jsonlite::fromJSON(opt_path, simplifyVector = TRUE)
dat <- utils::read.csv(csv_path, stringsAsFactors = FALSE, check.names = FALSE,
                       colClasses = "character", encoding = "UTF-8")
dat$TE <- as.numeric(dat$TE)
dat$seTE <- as.numeric(dat$seTE)
if (nrow(dat) < 2) die(sprintf("a forest needs at least two rows; %d were written", nrow(dat)))

# ---------------------------------------------------------------------------------- font
font_used <- function(want) {
  if (!nzchar(want)) return("sans")
  if (requireNamespace("systemfonts", quietly = TRUE)) {
    families <- tryCatch(systemfonts::system_fonts()$family, error = function(e) character())
    if (tolower(want) %in% tolower(families)) return(want)
    return("sans")
  }
  want
}

fontfamily_used <- font_used(if (is.null(opt$fontfamily)) "" else opt$fontfamily)

# ----------------------------------------------------------------------------- the model
subgroup_column <- if (is.null(opt$subgroup)) NULL else as.character(opt$subgroup)
subgroup <- NULL
if (!is.null(subgroup_column) && subgroup_column %in% names(dat)) {
  levels_seen <- opt$subgroup_levels
  if (is.null(levels_seen)) levels_seen <- unique(dat[[subgroup_column]])
  subgroup <- factor(dat[[subgroup_column]], levels = levels_seen)
}

m <- metagen(TE = TE, seTE = seTE, studlab = studlab, data = dat,
             sm = opt$sm, common = FALSE, random = TRUE,
             method.tau = opt$method_tau,
             method.random.ci = opt$method_random_ci,
             method.predict = opt$method_predict,
             level = opt$level, level.ma = opt$level, level.predict = opt$level,
             subgroup = subgroup,
             subgroup.name = if (is.null(subgroup)) NULL else "Line",
             title = if (is.null(opt$title)) "" else opt$title)

# meta refuses a prediction interval below k = 3 and reports df.predict as NA there. The df the
# METHOD uses at this k is still a fact about the call, and it is what the caller cross-checks
# against our own prediction interval, so it is reported with the source it came from.
implied_df <- function(method, k) {
  switch(toupper(method), HTS = k - 2, V = k - 1, S = Inf, R = k - 2, NA)
}
df_predict <- m$df.predict
df_predict_from <- "meta"
if (length(df_predict) != 1 || is.na(df_predict)) {
  df_predict <- implied_df(opt$method_predict, m$k)
  df_predict_from <- "method (meta reports NA below k = 3)"
}

one <- function(x) if (is.null(x) || length(x) == 0) NULL else unname(x[1])
named <- function(x) {
  if (is.null(x) || length(x) == 0) return(NULL)
  as.list(stats::setNames(as.numeric(x), names(x)))
}

# ------------------------------------------------------------------------------ the plot
sortvar <- if (identical(opt$sortvar, "TE")) m$TE else NULL

addline1 <- if (is.null(opt$text_addline1)) "" else opt$text_addline1
addline2 <- if (is.null(opt$text_addline2)) "" else opt$text_addline2
# forest.meta puts label.left/label.right on the axis's OWN row whenever a heterogeneity line and
# an additional line are both printed, so the direction labels land on top of the tick numbers.
# A leading line break gives them a row of their own; without an addline nothing is shifted.
label_break <- if (nzchar(addline1) || nzchar(addline2)) "\n" else ""
label_left <- paste0(label_break, opt$label_left)
label_right <- paste0(label_break, opt$label_right)
type_study <- if (is.null(opt$type_study)) "square" else as.character(opt$type_study)
col_square <- if (is.null(opt$col_square)) "gray40" else as.character(opt$col_square)
col_circle <- if (is.null(opt$col_circle)) "gray60" else as.character(opt$col_circle)

draw <- function() {
  meta::forest(m,
               sortvar = sortvar,
               prediction = isTRUE(opt$prediction),
               print.tau2 = isTRUE(opt$print_tau2),
               digits = as.integer(opt$digits),
               xlim = as.numeric(opt$xlim),
               fontfamily = fontfamily_used,
               weight.study = opt$weight_study,
               label.left = label_left,
               label.right = label_right,
               smlab = opt$smlab,
               leftcols = as.character(opt$leftcols),
               leftlabs = as.character(opt$leftlabs),
               rightcols = as.character(opt$rightcols),
               rightlabs = as.character(opt$rightlabs),
               plotwidth = opt$plotwidth,
               colgap.forest.left = opt$colgap_forest_left,
               colgap.forest.right = opt$colgap_forest_right,
               type.study = type_study,
               col.square = col_square,
               col.square.lines = col_square,
               col.circle = col_circle,
               col.circle.lines = col_circle,
               test.subgroup = !is.null(subgroup),
               # the subgroup rows are headed by the level itself ("Best guess"), not by
               # "Line = Best guess": the name of the grouping is an implementation detail here
               print.subgroup.name = FALSE,
               text.addline1 = addline1,
               text.addline2 = addline2)
}

.canopy$res <- if (is.null(opt$res)) 300 else as.numeric(opt$res)
.canopy$pointsize <- if (is.null(opt$pointsize)) 10 else as.numeric(opt$pointsize)
width_in <- if (is.null(opt$width_in)) 11 else as.numeric(opt$width_in)
height_in <- if (is.null(opt$height_in)) 6 else as.numeric(opt$height_in)

files <- list()
unavailable <- list()
for (format in as.character(opt$formats)) {
  problem <- device_error(format, width_in, height_in)
  if (nzchar(problem)) {
    unavailable[[format]] <- problem
    next
  }
  path <- paste0(out_stem, ".", format)
  written <- tryCatch({
    do.call(DEVICES[[format]], list(file = path, width = width_in, height = height_in))
    draw()
    grDevices::dev.off()
    TRUE
  }, error = function(e) {
    while (grDevices::dev.cur() > 1) grDevices::dev.off()
    unavailable[[format]] <<- conditionMessage(e)
    FALSE
  })
  if (isTRUE(written) && file.exists(path) && file.info(path)$size > 0)
    files[[format]] <- path
  else if (is.null(unavailable[[format]]))
    unavailable[[format]] <- sprintf("the %s device wrote nothing", format)
}

if (length(files) == 0)
  die(paste("no format could be written:",
            paste(sprintf("%s (%s)", names(unavailable), unlist(unavailable)), collapse = "; ")))

emit(list(
  ok = TRUE,
  k = m$k,
  TE_random = one(m$TE.random),
  lower_random = one(m$lower.random),
  upper_random = one(m$upper.random),
  tau2 = one(m$tau2),
  I2 = one(m$I2),
  Q = one(m$Q),
  Q_df = one(m$df.Q),
  pi_low = if (isTRUE(opt$prediction)) one(m$lower.predict) else NULL,
  pi_high = if (isTRUE(opt$prediction)) one(m$upper.predict) else NULL,
  df_predict = df_predict,
  df_predict_from = df_predict_from,
  subgroup_TE = named(m$TE.random.w),
  r_version = paste(R.version$major, R.version$minor, sep = "."),
  meta_version = as.character(packageVersion("meta")),
  fontfamily_used = fontfamily_used,
  files = files,
  unavailable = unavailable))

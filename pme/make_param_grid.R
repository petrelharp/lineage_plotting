#!/usr/bin/env Rscript

README <- " From https://github.com/petrelharp/lineage_plotting/issues/2

Our parameters are:

    θ: proportional to inverse difference in per-capita birth and death rates
    σ = 1/sqrt(θ): dispersal distance
    ε: interaction distance
    N: population density per unit area
    N_loc = ε^d N: neighborhood size (up to a constant)
    T = 1/θ: one unit of rescaled time, the time scale on which population dynamics happen

As we take θ to infinity, how should we change these? Let's fix the spatial scale to

    W = 40: width of the range

Let's aim to scale

    θ between 1 and 100 (where θ=1 probably has a difference between birth and death rates of around 0.1), so
    σ goes from 1 to 0.1

We'll have two conditions:

    (local): ε=σ, so from 1 to 0.1, or
    (nonlocal): ε=10σ, so from 10 to 1.

We also want to have θ/N going to either zero (deterministic) or a nonzero (random) limit, so:

    (deterministic): N = sqrt(θ)^3, so from 1 to 1000
    (random): N = θ, so from 1 to 100

In these four cases we have maximum neighborhood sizes of:

    (local, determinstic): N_loc = 100 (d=1) or = 10 (d=2)
    (local, random): N_loc = 10 (d=1) or = 1 (d=2)
    (nonlocal, deterministic): N_loc = 1000 (d=1 or d=2)
    (nonlocal, random): N_loc = 100 (d=1 or d=2)

"

library(jsonlite)

defaults <- read_json("example_params.json")
defaults$WIDTH <- 70

values <- expand.grid(
    THETA = c(1, 10, 100),
    DISPERSAL_SIGMA = c(0.1, 1),
    EPS_OVER_SIGMA = c(1, 10),
    K_POWER = c(1, 1.5)
)

values$EPSILON <- values$DISPERSAL_SIGMA * values$EPS_OVER_SIGMA
values$K <- values$THETA ^ values$K_POWER

stopifnot(all(6 * values$EPSILON < defaults$WIDTH))

outdir <- "param_grid"
dir.create(outdir, showWarnings=FALSE)

for (k in 1:nrow(values)) {
    subdir <- file.path(outdir, sprintf("sim_%04d", k))
    dir.create(subdir, showWarnings=FALSE)
    params <- defaults
    params <- params[setdiff(names(params), "README")]
    for (n in c("THETA", "DISPERSAL_SIGMA", "EPSILON", "K")) {
        stopifnot(n %in% names(params))
        params[[n]] <- values[[n]][k]
    }
    stopifnot(!file.exists(file.path(subdir, "params.json")))
    write_json(params, path=file.path(subdir, "params.json"))
}

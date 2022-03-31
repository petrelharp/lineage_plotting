library(jsonlite)

defaults <- list(
     DIMENSION = 2,
     THETA = 10,
     K = 20,
     MIN_DENSITY = 0.1,
     DT = 0.1,
     EPSILON = 1,
     DISPERSAL_SIGMA = 0.1,
     WIDTH = 6,
     HEIGHT = 6,
     INIT_PROP = 1.0,
     BURNIN = 400,
     RUNTIME = 400,
     NUM_SNAPSHOTS = 41,
)

# We want to take theta to infinity,
#  keeping theta/K small, (let's say 0.1?),
#  and keeping epsilon fixed.

pg <- data.frame(
            K = c(20, 60, 100, 150, 250)
)

pg$outdir <- sprintf("sim_K_%04d", pg$K)

for (k in 1:nrow(pg)) {
    dir.create(pg$outdir[k], showWarnings=FALSE)
    params <- defaults
    for (n in names(pg)) {
        if (n %in% names(params)) {
            params[[n]] <- pg[[n]][k]
        }
    }
    write_json(params, path=file.path(pg$outdir[k], "params.json"))
}

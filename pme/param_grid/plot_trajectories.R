#!/usr/bin/env Rscript

library(jsonlite)

simdirs <- list.files(".", "sim_.*")

logs <- do.call(rbind, lapply(simdirs, function (d) {
                   params <- data.frame(rbind(unlist(read_json(file.path(d, "params.json")))))
                   params$dir <- d
                   lfs <- lapply(list.files(d, "pme_.*.log", full.names=TRUE), function (x) cbind(params, read.csv(x)))
                   return(do.call(rbind, lfs))
}))
logs$dir <- factor(logs$dir, levels=simdirs)

Kvals <- sort(unique(logs$K))
Kcols <- rainbow(2*length(Kvals))

pdf(file="trajectories.pdf", width=6, height=4, pointsize=10)
layout(t(1:2))
for (log in c('', 'x')) {
    plot(0, 0, xlab='generation', ylab='pop size', type='n', xlim=range(logs$generation), ylim=range(logs$num_individuals), log=log)
    for (d in levels(logs$dir)) {
        with(subset(logs, dir == d), {
                iK <- match(unique(K), Kvals)
                lines(generation, num_individuals, type='l', col=Kcols[iK])
        })
    }
    legend("topright", lty=1, col=Kcols, legend=sprintf("K = %0.0f", Kvals))
}
dev.off()

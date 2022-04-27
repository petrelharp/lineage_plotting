import os, subprocess
import numpy as np
import matplotlib

from .spatial_slim_ts import *
from .animate import *
from .plot import *

def run_slim(script, seed = 23, 
             **kwargs):
    scriptbase = "_".join(script.split(".")[:-1])
    if not os.path.isdir(scriptbase):
        os.mkdir(scriptbase)
    kwstrings = [str(u) + "_" + str(kwargs[u]) for u in kwargs]
    base = os.path.join(scriptbase, 
            "_".join(["run"] + kwstrings + ["seed_" + str(seed)]))
    treefile = base + ".trees"
    if os.path.isfile(treefile):
        print(treefile, "already exists.")
    else:
        logfile = base + ".log"
        slim_command = ["slim", "-s {}".format(seed)]
        slim_command += ["-d {}={}".format(k, v) for k, v in kwargs.items()]
        slim_command += ["-d \"OUTPATH='{}'\"".format(treefile), script]
        print(" ".join(slim_command))
        with open(logfile, "w") as log:
            subprocess.call(" ".join(slim_command), shell=True, stdout=log)
        if not os.path.isfile(treefile):
            raise ValueError("SLiM failed to produce output file {}".format(treefile))
    return(treefile)

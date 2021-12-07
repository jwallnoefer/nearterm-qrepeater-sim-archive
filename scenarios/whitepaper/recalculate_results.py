import os, sys; sys.path.insert(0, os.path.abspath("."))
import numpy as np
import pandas as pd
from libs.aux_functions import standard_bipartite_evaluation


def recalculate(data_directory):
    c = 2e8
    data_series = pd.read_pickle(os.path.join(data_directory, "raw_data.bz2"))
    result_list = [standard_bipartite_evaluation(data_frame=df, trial_time=length / c) for length, df in data_series.items()]
    output_data = pd.DataFrame(data=result_list, index=data_series.index, columns=["fidelity", "fidelity_std", "key_per_time", "key_per_time_std", "key_per_resource", "key_per_resource_std"])
    output_data.to_csv(os.path.join(data_directory, "result.csv"))


if __name__ == "__main__":
    base_dir = os.path.join("results", "whitepaper_timeslots")
    for subdir, dirs, files in os.walk(base_dir):
        for file in files:
            if file == "raw_data.bz2":
                recalculate(subdir)

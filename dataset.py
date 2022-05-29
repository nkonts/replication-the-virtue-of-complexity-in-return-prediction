import pandas as pd
import numpy as np


# see footnote [22]
COLUMNS = ["b/m", "de", "dfr", "dfy", "dp", "dy", "ep", "infl", "ltr", "lty", "ntis", "svar", "tbl", "tms", "returns"]


def load_nber():
    nber = pd.read_csv("data/NBER_20210719_cycle_dates_pasted.csv")[1:]
    nber["peak"] = pd.to_datetime(nber["peak"])
    nber["trough"] = pd.to_datetime(nber["trough"])
    return nber

def load_data():
    data_raw = pd.read_csv("data/PredictorData2021 - Monthly.csv")
    data_raw["yyyymm"] = pd.to_datetime(data_raw["yyyymm"], format='%Y%m', errors='coerce')
    data_raw["Index"] = data_raw["Index"].str.replace(",", "")
    data_raw = data_raw.set_index("yyyymm")
    data_raw[data_raw.columns] = data_raw[data_raw.columns].astype(float)
    data_raw = data_raw.rename({"Index":"prices"}, axis=1)

    # Calculate missing columns according to the explaination in m Welch and Goyal (2008) 
    data_raw["dfy"] = data_raw["BAA"] - data_raw["AAA"]
    data_raw["tms"] = data_raw["lty"] - data_raw["tbl"]
    data_raw["de"] = np.log(data_raw["D12"]) - np.log(data_raw["E12"])
    data_raw["dfr"] = data_raw["corpr"] - data_raw["ltr"]
    data_raw["lag_price"] = data_raw["prices"].shift()
    data_raw["dp"] = np.log(data_raw["D12"]) - np.log(data_raw["prices"])
    data_raw["dy"] = np.log(data_raw["D12"]) - np.log(data_raw["lag_price"])
    data_raw["ep"] = np.log(data_raw["E12"])  - np.log(data_raw["prices"])

    data_raw["returns"] = data_raw["prices"].pct_change() # Maybe use CRSP_SPvw - Value weighted return?
    returns = data_raw["returns"].copy()

    data = data_raw[COLUMNS].dropna()
    returns = returns[returns.index.isin(data.index)]

    return data, returns
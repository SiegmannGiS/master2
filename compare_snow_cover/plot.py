import matplotlib
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
plt.style.use('ggplot')
from compare_snow_cover import *


location = "astental"

# read data
datadict = {}
for i, element in enumerate(os.listdir("data")):
    if element.split("_")[1] == location:
        dictname = "_".join(element.split("_")[2:])[:-4]
        print dictname
        datadict[dictname] = pd.read_csv(os.path.join("data", element), index_col="date", parse_dates="date")



# plot
fig = plt.figure(figsize=(15,5))

# Iteration
for i, statname in enumerate(["ACC", "CSI", "BIAS"]):
    num = 131 + i
    ax = plt.subplot(num)
    ax.set_title(statname, y=1.05)
    for key,df in datadict.iteritems():
        if "landsat" in key:
            plt.plot(df[statname], marker="^", color="#fcad04", linestyle="None", markersize=9)
        else:
            df_rolling = pd.rolling_mean(df[statname], 30, center=True)
            plt.plot(df[statname], marker=".", color="#f76050", linestyle="None")
            plt.plot(df_rolling, color="#2e62cc")

        if statname != "BIAS":
            plt.ylim((0,1))

        plt.xticks(rotation=30)

plt.tight_layout()
plt.savefig("C:\Master\Images thesis/auswertung_stat_%s.png" %location)
plt.show()
plt.close()
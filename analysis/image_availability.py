import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import os
plt.style.use('ggplot')

imagepath = "C:\Master\images/vernagtferner14-16"


files = os.listdir(imagepath)
imagedate = [datetime.strptime("-".join(x.split("_")[:2]), "%Y-%m-%d-%H-%M") for x in files]
images_ts = pd.Series(1,imagedate)

df = pd.DataFrame(images_ts, columns=["Vernagferner"])

df = df.resample("D", how="sum")["20141001":"20150430"]

# Comparison Landsat 8 OLI
lpath = "C:\Master\settings/vernagtferner14-16\landsat"
landsat10 = pd.read_csv(os.path.join(lpath,"LANDSAT_8_115275_10%.csv"), sep=",", parse_dates="Date Acquired", index_col="Date Acquired")
landsat10["Availibility"] = 1
landsat10 = landsat10["Availibility"]
landsat10 = landsat10.resample("D", how="sum")

landsat50 = pd.read_csv(os.path.join(lpath,"LANDSAT_8_115276_50%.csv"), sep=",", parse_dates="Date Acquired", index_col="Date Acquired")
landsat50["Availibility"] = 1
landsat50 = landsat50["Availibility"]
landsat50 = landsat50.resample("D", how="sum")




ax = plt.subplot(111)
plt.rc('figure', figsize=(11.69,8.27))
ax.bar(df.index,df.values, label="Webcam images Vernagtferner")
ax.bar(landsat50.index,landsat50.values, color="#f87c7c", alpha=0.5, label="Landsat 8 OLI, cloud cover < 50%")
ax.bar(landsat10.index,landsat10.values, color="#fa0707", label="Landsat 8 OLI, cloud cover < 10%")

ax.xaxis_date()
ax.set_ylim([0,4])
plt.ylabel("Images per day")
plt.xlabel("Date")
plt.title("Image availability 10/2014 - 04/2015", y=1.04)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("C:\Master\Images thesis\Image availibility.jpg")
plt.show()
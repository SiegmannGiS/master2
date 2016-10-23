import os
import tarfile

path = "/home/marcel/master/landsat/downloads"

for folder in os.listdir("/home/marcel/master/landsat/downloads"):
    actual_path = os.path.join(path,folder)
    for element in os.listdir(actual_path):
        if element[-4:] == ".tgz":
            print(element)
            band = element[:-4] + "_BQA.TIF"
            if not os.path.exists(os.path.join(actual_path,band)):
                with tarfile.open(os.path.join(actual_path,element), mode="r:gz") as file:
                    file.extract(band, path=actual_path)

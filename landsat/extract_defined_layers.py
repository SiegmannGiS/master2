import os

path = "/home/marcel/master/landsat/downloads"
path2 = "/home/marcel/master/landsat/layers"
program = "/home/marcel/master/landsat/scripte/Landsat_LDOPE/linux64bit_bin/unpack_oli_qa"
for folder in os.listdir(path):
    actual_path = os.path.join(path,folder)
    for element in os.listdir(actual_path):
        if element[-4:] == ".TIF":
            print(element)
            cmd = "%s --ifile=%s --ofile=%s --snow_ice=med --cloud=med --cirrus=med" \
                  %(program, os.path.join(actual_path,element), os.path.join(path2,element.split("_")[0]))
            os.system(cmd)
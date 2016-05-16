import os

path = "E:\Master\Practise\PRACTISE_Matlabv2\Config_Vernagtferner\slr_ufs\Export_Output.txt"
path2 = "E:\Master\Practise\PRACTISE_Matlabv2\Config_Vernagtferner\slr_ufs/vernagt.gcp.txt"
with open(path, "r") as fobj:
    with open(path2, "w") as fobj_out:
        fobj_out.write("POINT_X\tPOINT_Y\tPOINT_Z\tPIXEL_COL\tPIXEL_ROW\tGCPname\n")
        lines = fobj.readlines()
        for i,line in enumerate(lines):
            if i > 0:
                line = line.strip().split(",")
                line = "\t".join(line[1:3]+[line[6]]+line[3:5]+[line[5]])
                print line
                fobj_out.write("%s\n" %line)
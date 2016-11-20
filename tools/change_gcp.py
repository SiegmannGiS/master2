import os

location = "Vernagtferner"
path = "E:\Master\Practise\PRACTISE_Matlabv2\Config_%s\slr_ufs\Export_Output.txt" %location
path2 = "E:\Master\Practise\PRACTISE_Matlabv2\Config_%s\slr_ufs/%s.gcp.txt" %(location,location)
with open(path, "r") as fobj:
    with open(path2, "w") as fobj_out:
        fobj_out.write("POINT_X\tPOINT_Y\tPOINT_Z\tPIXEL_COL\tPIXEL_ROW\tGCPname\n")
        lines = fobj.readlines()
        for i,line in enumerate(lines):
            if i > 0:
                line = line.strip().split(",")
                line[3] = str(float(line[3]) + float(line[-1]))
                line = "\t".join(line[1:-1]) #line[1:3]+[line[6]]+line[3:5]+[line[5]]
                print(line)
                fobj_out.write("%s\n" %line)
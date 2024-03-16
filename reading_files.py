import csv
import os

all_mofs={}


os.chdir('/home/yushanc2/Documents/fromXY/from_xy_pc/ZEOPP_calculations/full')
with open('zeopp.csv','r') as f:
    csvfile=csv.reader(f,delimiter=',')  

    # next(csvfile)
    i=0
    for row in csvfile:
        
        if i==0:
            print(row)
        else:
            all_mofs[row[0]]={}
            work_dict=all_mofs[row[0]]
        
        i+=1
        










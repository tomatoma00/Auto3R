import os
import numpy as np
import argparse
parser = argparse.ArgumentParser(description="Train or render 3D models using uid(s).")
parser.add_argument('--uid', type=str)
parser.add_argument('--repeat', type=int)
parser.add_argument('--folder', type=str, default = "normalexp")
args = parser.parse_args()
uid = args.uid
wsavepath = f"{args.folder}/{uid}/worstpnsr{args.repeat}.txt"
savepath = f"{args.folder}/{uid}/avepnsr{args.repeat}.txt"
addlist = []
wavglist = []
avglist = []

numberiter = [399, 899, 1499, 2199, 2999, 3899, 4899, 5999, 7199, 8499, 9899, 11399, 12999, 14699, 16499, 18399, 21000, \
           26000, 30000]

for i in range(len(numberiter)):
    iternum = numberiter[i]
    psnrpath = f"{args.folder}/{uid}/train/ours_{iternum}/psnr.txt"
    try:
        aa = np.loadtxt(psnrpath)
    except:
        print("pass",iternum) 
        continue
    smallest5 = np.partition(aa, 5)[:5]
    wavg = smallest5.mean()
    avg = np.array(aa).mean()
    print(avg,wavg)
    addlist.append(iternum)
    wavglist.append(wavg)
    avglist.append(avg)
    
with open(wsavepath,'w') as f:
    for i in range(len(avglist)):
        f.write(f"{addlist[i]:03d} {wavglist[i]}\n")

with open(savepath,'w') as f:
    for i in range(len(avglist)):
        f.write(f"{addlist[i]:03d} {avglist[i]}\n")

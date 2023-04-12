import pandas as pd
import os
import glob
import re

def find_date(x):
    sec_occ = x.index("_", x.index("_") + 1, len(x))  # str.index(str, beg=0, end=len(string))
    thi_occ = x.index("_", sec_occ + 1, len(x))
    return x[sec_occ + 1:thi_occ]


files_dict = {"textual data": [], "label": [], "time": []}

# Step 1: get a list of all txt files in target directory
my_dir = './HKEX Reports/Fraudulent'
filelist = []
filesList = []
os.chdir(my_dir)

# Step 2: Build up list of files:
for files in glob.glob("*ENG.txt"):
    fileName, fileExtension = os.path.splitext(files)
    filelist.append(fileName)  # filename without extension
    filesList.append(files)  # filename with extension

for f in filesList:
    fo = open(f, 'r', encoding='utf-8')
    count = 0

    text = ""
    for line in fo.readlines():  # read each line
        line = re.sub('[^a-zA-Z0-9\n\.]', ' ', line)
        line = line.strip()
        text = text + line
    files_dict["textual data"].append(text)
    files_dict["label"].append("F")
    files_dict["time"].append(find_date(f))
    fo.close()

fraud_df = pd.DataFrame(files_dict)

fraud_df.to_csv("2.csv", encoding='utf-8', index=False)

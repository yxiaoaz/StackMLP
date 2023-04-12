import pandas as pd
import os
import glob
import re

def find_date(x):
    sec_occ = x.index("_", x.index("_") + 1, len(x))  # str.index(str, beg=0, end=len(string))
    thi_occ = x.index("_", sec_occ + 1, len(x))
    return x[sec_occ + 1:thi_occ]

my_dir = "./HKEX Reports/Fraudulent/missing_files"
filelist = []
filesList = []
os.chdir(my_dir)

# Step 2: Build up list of files:
for files in glob.glob("*.txt"):
    fileName, fileExtension = os.path.splitext(files)
    filelist.append(fileName)  # filename without extension
    filesList.append(files)  # filename with extension

# file1 = open('myfile.txt', 'r')

files_dict = {"textual data": [], "label": [], "time": []}

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
    files_dict["time"].append(f[0:8])  # e.g. 20200517.txt, strip ".txt"
    fo.close()

fraud_df_2 = pd.DataFrame.from_dict(files_dict)
fraud_df_2.to_csv("3.csv",encoding='utf-8',index=False)
print(fraud_df_2)

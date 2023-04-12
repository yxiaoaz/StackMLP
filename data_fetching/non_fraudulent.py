import os
import pandas as pd
import glob
import re

files_dict = {"textual data": [], "label": [], "time": []}

# assign directory
directory = ['./HKEX Reports/hkex_reports_annual/hkex_reports_mda_text',
             './HKEX Reports/hkex_reports_semi-annual/hkex_reports_mda_text']

def find_(num,s,target_num,start=0):  # find the index of "_" in pathfile.
    if num==target_num:
        return s.index("_",start)
    start=s.index("_",start)
    return find_(num+1,s,target_num,start+1)


i=0

for dir in directory:
    for subdir, dirs, Files in os.walk(dir):
        for File in Files:
            # print os.path.join(subdir, file)
            filepath = subdir + os.sep + File
            '''
            File:00178_523961_20070711_Annual-Report-20062007_pages_n-n_ManagementDiscussionandAnalysis_ENG.txt
            (original)subdir:./HKEX Reports/HKEX Reports/hkex_reports_annual/hkex_reports_mda_text\2007_07
            '''
            #print(filepath)

            if(filepath.endswith("ENG.txt")):

                try:
                    fo = open(filepath, 'r', encoding='utf-8')
                    count = 0

                    text = ""
                    for line in fo.readlines():
                        line = re.sub('[^a-zA-Z0-9\n\.]', ' ', line)
                        line = line.strip()
                        text = text + line
                    files_dict["textual data"].append(text)
                    files_dict["label"].append("NF")
                    files_dict["time"].append(filepath[find_(0,filepath,target_num=7)+1:find_(0,filepath,target_num=8)])
                    fo.close()
                except:
                    i=i+1

print(i)
non_fraud_df = pd.DataFrame(files_dict)
non_fraud_df.to_csv("1.csv",encoding='utf-8',index=False)

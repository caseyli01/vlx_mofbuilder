import re
import glob
import os
from itp_process import parsetop

def itp_extract(itp_file):
    with open(itp_file, "r") as f:
        lines = f.readlines()
    keyword1 = "atomtypes"
    keyword2 = "moleculetype"
    for eachline in lines:  # search for keywords and get linenumber
        if re.search(keyword1, eachline):
            start = lines.index(eachline) + 2
            
        elif re.search(keyword2, eachline):
            end = lines.index(eachline) - 1

    target_lines = [line for line in lines[start:end] if line.strip()]

    newstart = end + 1

    with open(itp_file, "w") as fp:
        fp.writelines(lines[newstart:])

    #target_lines.append("\n")
    sec1 = target_lines
    return sec1

def extract_atomstypes(itp_path):
    all_secs = []
    for f in glob.glob(os.path.join(itp_path,'*itp')):
        if os.path.basename(f) not in ['posre.itp']:
            print(f)
            sec_atomtypes = itp_extract(f)
            all_secs+=sec_atomtypes
    return all_secs

def get_unique_atomtypes(all_secs):
    types = [str(line.split()[0]) for line in all_secs]
    overlap_lines = []
    for ty in set(types):
        search = [ind for ind,value in enumerate(types) if value == ty]
        if len(search)>1:
            overlap_lines+=search[1:]
    unique_atomtypes = [all_secs[i] for i in range(len(all_secs)) if i not in overlap_lines]
    return unique_atomtypes

def genrate_top_file(itp_path,data_path,res_info,model_name):
    all_secs = extract_atomstypes(itp_path)
    unique_atomtypes = get_unique_atomtypes(all_secs)
    middlelines, sectorname = parsetop(data_path+'/nodes_itps/template.top') #fetch template.top

    top_res_lines=[]
    for resname in list(res_info):
        line = "%-5s%16d" % (resname[:3],res_info[resname])
        top_res_lines.append(line)
        top_res_lines.append("\n")

    top_itp_lines = []
    for i in glob.glob(os.path.join(itp_path,'*itp')):
        if os.path.basename(i) not in ['posre.itp']:
            line = '#include "itps/'+os.path.basename(i)+'"'+'\n'
            top_itp_lines.append(line)
    sec1 = unique_atomtypes
    sec2 = top_itp_lines
    sec3 = ["MOF" + "\n" + "\n"]
    sec4 = top_res_lines+ ["\n"]+["\n"]

    newtop = (
        middlelines[0]
        + ["\n"]
        + ["\n"]
        + middlelines[1]
        + unique_atomtypes
        + ["\n"]
        + ["\n"]
        + top_itp_lines
        + ["\n"]
        + ["\n"]
        + middlelines[2]
        + ["MOF" ]
        + ["\n"]
        + ["\n"]
        + middlelines[3]
        + ["\n"]
        + top_res_lines

    )
    topname = model_name+'.top'
    top_path = 'MD_run/'+topname
    with open(top_path, "w") as f:
            f.writelines(newtop)
    print(topname,'is generated')
    return top_path

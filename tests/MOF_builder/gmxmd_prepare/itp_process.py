import re
import os
import glob


def parsetop(inputfile):
    # newpath = os.path.abspath ( '')+'/'+str(outputfile)+'/'    # input file
    # os.makedirs(newpath,exist_ok=True)
    with open(inputfile, "r") as fp:
        original_lines = fp.readlines()

    lines = [line for line in original_lines if line.strip()]
    number = []
    includelines_number = []
    lineNumber = 1
    lineNumber_include = 1
    keyword1 = "]"  # input('Keyword:')
    for eachline in lines:  # search for keywords and get linenumber
        m = re.search(keyword1, eachline)
        if m is not None:
            number.append(lineNumber - 1)  # split by linenumber
        lineNumber += 1

    #keyword2 = "#include"  # input('Keyword:')
    #for eachline in lines:  # search for keywords and get linenumber
    #    n = re.search(keyword2, eachline)
    #    if n is not None:
    #        includelines_number.append(lineNumber_include - 1)  # split by linenumber
    #    lineNumber_include += 1
    #includelines_number.sort()
    #number.append(includelines_number[0])

    number.append(len(lines))
    number = list(set(number))
    number.sort()
    size = int(len(number))
    # print(number)

    middlelines = []
    sectorname = []
    for i in range(size - 1):
        # set output range
        start = number[i]
        end = number[i + 1]
        middlelines.append(lines[start:end])
        sectorname.append(lines[start])

        # fp_w = open(sectorname+'.txt','w')
        # for key in middlelines:
        #    fp_w.write(key)
        # fp_w.close()
    return middlelines, sectorname

# fetch atomtype sector
#for i in range(len(subfolders)):
def top_gen(itp_file,middlelines,outtopname,input_res_num):
    with open(itp_file, "r") as f:
        lines = f.readlines()

    keyword1 = "atomtype"
    keyword2 = "moleculetype"
    for eachline in lines:  # search for keywords and get linenumber
        if re.search(keyword1, eachline):
            start = lines.index(eachline) + 2
        elif re.search(keyword2, eachline):
            end = lines.index(eachline) - 1

    target_lines = [line for line in lines[start:end] if line.strip()]

    newstart = end + 1

    with open("linker.itp", "w") as fp:
        fp.writelines(lines[newstart:])

    target_lines.append("\n")
    sec1 = target_lines
    sec2 = [
        '#include "../itps/linker.itp"'
        + "\n"
        + '#include "../itps/tip3p.itp"'
        + "\n"
        + '#include "../itps/K.itp"'
        + "\n"
        + "\n"
    ]
    sec3 = ["MOF" + "\n" + "\n"]
    sec4 = [

        "\n"+input_res_num
        + "\n"
        + "\n"
    ]
    newtop = (
        middlelines[0]
        + middlelines[1]
        + sec1
        + middlelines[2]
        + sec2
        + middlelines[3]
        + sec3
        + middlelines[4]
        + sec4
    )


    #newpath = os.path.abspath ( '')+'/MDdemo/mdfiles/'   # input file
    #os.makedirs(newpath,exist_ok=True)

    with open(outtopname, "w") as f:
        f.writelines(newtop)
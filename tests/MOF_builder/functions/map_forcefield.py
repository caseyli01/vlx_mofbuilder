import re
import pandas as pd
import os 
import linecache

def parseff(INPUT):
    newpath = os.path.abspath ('')+'/Residues/parsedfile/'      # input file
    os.makedirs(newpath,exist_ok=True) 
    inputfile = str(INPUT)
    #outputfile = 'new'+inputfile.strip(".ff")
    #print(inputfile,outputfile+'.ff')
    #newpath = os.path.abspath ( '')+'/'+str(outputfile)+'/'    # input file
    #os.makedirs(newpath,exist_ok=True) 
    fp = open(inputfile,'r')
    number = []
    lineNumber = 1
    keyword = "]"                                   #input('Keyword:')
    for eachline in fp:                                    #search for keywords and get linenumber
        m = re.search(keyword, eachline)
        if m is not None:
            number.append(lineNumber-1)                     #split by linenumber
        lineNumber+=1 
    number.append(len(open(INPUT).readlines()))
    number = list(set(number))
    number.sort()
    size = int(len(number))
    #print(number)
    for i in range(size-1):
                    #set output range
        start = number[i]
        end =  number[i+1]
        middlelines = linecache.getlines(inputfile)[start:end]
        section = re.findall(r'\[(.*?)\]', middlelines[0])
        title=section[0].split()[0]
        if title =='dihedrals':
            if 'impropers' in middlelines[1]:
                title = 'dihedrals_im'
        print(title)
        fp_w = open(newpath+title+'.txt','w')
        for key in middlelines:
            fp_w.write(key)
        fp_w.close()
    return newpath



def readgro(input):
    with open(input, "r") as f:
        lines = f.readlines()
        number_resname = lines[0].split()[0]
        resname = re.sub(r"^\d+", "", number_resname)
        atoms = []
        for i in range(len(lines)):
            atoms.append(lines[i].split()[1])
    return resname, atoms

def readxyz(input):
    with open(input, "r") as f:
        lines = f.readlines()
        #number_resname = lines[0].split()[0]
        resname = os.path.basename(input).removesuffix('.xyz')#re.sub(r"^\d+", "", number_resname)
        atoms = []
        for i in range(2,len(lines)):
            if len(lines[i].split())==0:
                continue
            atoms.append(lines[i].split()[0]+str(i-1))
    return resname,atoms

def getatomtype(input):
    with open(input, "r") as f:
        lines = f.readlines()

    return lines

def getmoleculetype(input, new_resname):
    with open(input, "r") as f:
        lines = f.readlines()

    #with open("moleculetype.txt", "w") as f:
    newff = []
    newff.append(lines[0])
    newff.append(lines[1])
    values = lines[2].split()
    values[0] = new_resname
    formatted_line = "%-7s%7s" % (values[0], values[1])
    newff.append(formatted_line + "\n")
        #f.writelines(newff)
    return newff

def getatoms(input,father,son,son_atoms,son_resname):
    with open(input, "r") as f:
        lines = f.readlines()

    with open(os.path.abspath ('')+'/Residues/parsedfile/'+"atoms1", "w") as f:
        newff = []
        for i in range(2, len(lines)):
            values = lines[i].split()
            #print(values)
            if len(values)==0:
                continue
            # print(son)
            res_index = father.index(values[0])
            values[0] = son[res_index]
            values[6] = float(values[6])
            values[7] = float(values[7])
            if len(values)>8:
                values[10] = float(values[10])

                formatted_line = "%7s%7s%7s%7s%7s%7s%15.8f%15.6f%7s%7s%15.6f" % (
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    values[6],
                    values[7],
                    values[8],
                    values[9],
                    values[10],
                )
                newff.append(formatted_line + "\n")
            else:
                formatted_line = "%7s%7s%7s%7s%7s%7s%15.8f%15.6f" % (
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    values[6],
                    values[7],
                )
                newff.append(formatted_line + "\n")


        f.writelines(newff)

    df = pd.read_csv(
        os.path.abspath ('')+'/Residues/parsedfile/'+"atoms1",
        sep='\s+',
        names=[
            "nr",
            "type",
            "resnr",
            "residue",
            "atom",
            "cgnr",
            "charge",
            "mass",
            "typeB",
            "chargeB",
            "massB",
        ],
    )
    sondf = df.sort_values(by="nr").reset_index(drop=True)
    sondf["residue"] = son_resname
    sondf["atom"] = son_atoms
    sondf["cgnr"] = sondf.index + 1
    sondf.loc[0, "massB"] = sondf.loc[0, "charge"]
    # print(sondf)
    for i in range(1, sondf.shape[0]):
        sondf.loc[i, "massB"] = sondf.loc[i - 1, "massB"] + sondf.loc[i, "charge"]

    sondf.to_csv(os.path.abspath ('')+'/Residues/parsedfile/'+"atoms2", sep="\t", header=None, index=False)

    with open(os.path.abspath ('')+'/Residues/parsedfile/'+"atoms2", "r") as f:
        sonlines = f.readlines()

    #with open("atoms.txt", "w") as f:
        newff = []
        newff.append(lines[0])
        newff.append(lines[1])
        # print(len(sonlines))
        for i in range(len(sonlines)):
            values = sonlines[i].split()
            # print(values)
            values[6] = float(values[6])
            values[7] = float(values[7])
            if len(values)>9:
                values[10] = float(values[10])

                formatted_line = "%6s%5s%7s%6s%6s%5s%13.6f%13.5f%2s%5s%10.6f" % (
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    values[6],
                    values[7],
                    values[8],
                    values[9],
                    values[10],
                )
                newff.append(formatted_line + "\n")
            else:
                formatted_line = "%6s%5s%7s%6s%6s%5s%13.6f%13.5f" % (
                    values[0],
                    values[1],
                    values[2],
                    values[3],
                    values[4],
                    values[5],
                    values[6],
                    values[7],
                )
                newff.append(formatted_line + "\n")

        #f.writelines(newff)
    os.remove(os.path.abspath ('')+'/Residues/parsedfile/'+"atoms1")
    os.remove(os.path.abspath ('')+'/Residues/parsedfile/'+"atoms2")
    return newff

def getbonds(input,father,son):
    with open(input, "r") as f:
        lines = f.readlines()

    #with open("bonds.txt", "w") as f:
        newff = []
        newff.append(lines[0])
        newff.append(lines[1])
        for i in range(2, len(lines)):
            values = lines[i].split()
            if len(values)==0:
                continue
            ai_index = father.index(values[0])
            values[0] = son[ai_index]
            aj_index = father.index(values[1])
            values[1] = son[aj_index]
            values[3] = float(values[3])
            values[4] = float(values[4])

            formatted_line = "%7s%7s%6s%15.7f%15.6f" % (
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
            )
            newff.append(formatted_line + "\n")
        #f.writelines(newff)
        return newff

def getpairs(input,father,son):
    with open(input, "r") as f:
        lines = f.readlines()

    #with open("pairs.txt", "w") as f:
        newff = []
        newff.append(lines[0])
        newff.append(lines[1])
        for i in range(2, len(lines)):
            values = lines[i].split()
            if len(values)==0:
                continue
            ai_index = father.index(values[0])
            values[0] = son[ai_index]
            aj_index = father.index(values[1])
            values[1] = son[aj_index]

            formatted_line = "%7s%7s%6s" % (values[0], values[1], values[2])
            newff.append(formatted_line + "\n")
        #f.writelines(newff)
        return newff

def getangles(input,father,son):
    with open(input, "r") as f:
        lines = f.readlines()

    #with open("angles.txt", "w") as f:
        newff = []
        newff.append(lines[0])
        newff.append(lines[1])
        for i in range(2, len(lines)):
            values = lines[i].split()
            if len(values)==0:
                continue
            ai_index = father.index(values[0])
            values[0] = son[ai_index]
            aj_index = father.index(values[1])
            values[1] = son[aj_index]
            ak_index = father.index(values[2])
            values[2] = son[ak_index]

            values[4] = float(values[4])
            values[5] = float(values[5])

            formatted_line = "%7s%7s%7s%6s%13.7f%12.6f" % (
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
            )
            newff.append(formatted_line + "\n")
        #f.writelines(newff)
        return newff

def getdihedrals(input,father,son):
    with open(input, "r") as f:
        lines = f.readlines()

    #with open("dihedrals.txt", "w") as f:
        newff = []
        newff.append(lines[0])
        newff.append(lines[1])
        newff.append(lines[2])
        for i in range(3, len(lines)):
            values = lines[i].split()
            if len(values)==0:
                continue
            ai_index = father.index(values[0])
            values[0] = son[ai_index]
            aj_index = father.index(values[1])
            values[1] = son[aj_index]
            ak_index = father.index(values[2])
            values[2] = son[ak_index]
            al_index = father.index(values[3])
            values[3] = son[al_index]

            values[5] = float(values[5])
            values[6] = float(values[6])

            formatted_line = "%7s%7s%7s%7s%6s%13.7f%12.7f%3s" % (
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
            )
            newff.append(formatted_line + "\n")
        #f.writelines(newff)
        return newff

def getdihedrals_im(input,father,son):
    with open(input, "r") as f:
        lines = f.readlines()

    #with open("dihedrals_im.txt", "w") as f:
        newff = []
        newff.append(lines[0])
        newff.append(lines[1])
        newff.append(lines[2])
        for i in range(3, len(lines)):
            values = lines[i].split()
            if len(values)==0:
                continue
            ai_index = father.index(values[0])
            values[0] = son[ai_index]
            aj_index = father.index(values[1])
            values[1] = son[aj_index]
            ak_index = father.index(values[2])
            values[2] = son[ak_index]
            al_index = father.index(values[3])
            values[3] = son[al_index]

            values[5] = float(values[5])
            values[6] = float(values[6])

            formatted_line = "%7s%7s%7s%7s%6s%13.7f%12.7f%3s" % (
                values[0],
                values[1],
                values[2],
                values[3],
                values[4],
                values[5],
                values[6],
                values[7],
            )
            newff.append(formatted_line + "\n")
        #f.writelines(newff)
        return newff

def map_forcefield_by_xyz(path,map_path,ff_xyz, new_xyz='EDGE.xyz'):
    mapfile = map_path
    moleculetype = path + "moleculetype.txt"
    atomtype = path + "atomtypes.txt"
    atom = path + "atoms.txt"
    bond = path + "bonds.txt"
    pair = path + "pairs.txt"
    angle = path + "angles.txt"
    dihedral = path + "dihedrals.txt"
    dihedral_im = path + "dihedrals_im.txt"

    fatherxyz = ff_xyz
    sonxyz = os.path.abspath ('')+'/Residues/'+new_xyz
    new_itp = os.path.abspath ('')+'/Residues/'+new_xyz.removesuffix('.xyz')+'.itp'

    with open(mapfile, "r") as f:
        father = []
        son = []
        lines = f.readlines()
        for i in range(len(lines)):
            values = lines[i].strip('\n').split()
            if len(values)==0:
                continue
            father.append(values[0])
            son.append(values[1])


    new_resname = new_xyz.removesuffix('.xyz')[:3]
    son_resname, son_atoms = readxyz(sonxyz)
    father_resname, father_atoms = readxyz(fatherxyz)

    atomtypes = getatomtype(atomtype)
    moleculetypes = getmoleculetype(moleculetype, new_resname)
    atoms = getatoms(atom,father,son,son_atoms,new_resname)
    if os.path.exists(bond):
        bonds = getbonds(bond,father,son) 
    else:
        bonds = False
    if os.path.exists(pair):
        pairs = getpairs(pair,father,son) 
    else:
        pairs = False
    if os.path.exists(angle):
        angles = getangles(angle,father,son) 
    else:
        angles = False
    if os.path.exists(dihedral):
        dihedrals = getdihedrals(dihedral,father,son)
    else:
        dihedrals = False
    if os.path.exists(dihedral_im):
        dihedrals_im = getdihedrals_im(dihedral_im,father,son)
    else:
        dihedrals_im = False

    with open(new_itp, "w") as f:
        f.write(";generated by veloxchem, mapped by mof_builder")
        f.write("\n")
        f.write("\n")
        f.writelines(atomtypes)
        f.write("\n")
        f.writelines(moleculetypes)
        f.write("\n")
        f.writelines(atoms)
        f.write("\n")
        if bonds:
            f.writelines(bonds) 
            f.write("\n")
        if pairs:
            f.writelines(pairs) 
            f.write("\n")
        if angles:
            f.writelines(angles) 
            f.write("\n")
        if dihedrals:
            f.writelines(dihedrals) 
            f.write("\n")
        if dihedrals_im:
            f.writelines(dihedrals_im) 
            f.write("\n")
    print(str(new_itp)+ ' is generated')
import numpy as np
import os
import ray
import psutil
import re

def __obj2off(objpath, offpath):
    '''
    将obj文件转换为off文件
    :param objpath: .obj文件的路径
    :param offpath: .off文件的路径的保存地址
    :return: 无
    '''
    line = ""

    vset = []
    fset = []
    with open(objpath,'r') as f:
        lines = f.readlines()
    p = re.compile(r'/+')
    space = re.compile(r' +')

    for line in lines:
        #拿到obj文件中一行，作为一个字符串
        tailMark = " "
        line = line+tailMark
        if line[0]!='v' and line[0]!='f' :
            continue

        parameters = space.split(line.strip())
        if parameters[0] == "v":   #如果是顶点的话
                Point = []
                Point.append(eval( parameters[1]) )
                Point.append(eval( parameters[2]) )
                Point.append(eval( parameters[3]) )
                vset.append(Point)

        elif parameters[0] == "f":   #如果是面的话，存放顶点的索引
                vIndexSets = []          #临时存放点的集合
                for i in range(1,len(parameters) ):
                    x = parameters[i]
                    ans = p.split(x)[0]
                    index = eval(ans)
                    index -= 1          #因为顶点索引在obj文件中是从1开始的，而我们存放的顶点是从0开始的，因此要减1
                    vIndexSets.append(index)

                fset.append(vIndexSets)

    with open(offpath, 'w') as out:
        out = open(offpath, 'w')
        out.write("OFF\n")
        out.write(str(vset.__len__()) + " " + str(fset.__len__()) + " 0\n")
        for j in range(len(vset)):
            out.write(str(vset[j][0]) + " " + str(vset[j][1]) + " " + str(vset[j][2]) + "\n")

        for i in range(len(fset)):
            s = str(len( fset[i] ))
            for j in range( len( fset[i] ) ):
                s = s+ " "+ str(fset[i][j])
            s += "\n"
            out.write(s)

    print("{} 转换成 {} 成功.".format( p.split(objpath)[-1], p.split(offpath)[-1] ))


def convertObj2Off(objRootDir, dstRootDir):
    """Convert obj files to off files"""
    if not os.path.exists(dstRootDir):
        os.mkdir(dstRootDir)
    firstSubDirFlag = True
    for root, dirs, fs in os.walk(objRootDir):
        if firstSubDirFlag: #第一级目录
            dstSubDirs = [os.path.join(dstRootDir, dir) for dir in dirs]
            for dstSubDir in dstSubDirs:
                if not os.path.exists(dstSubDir):
                    os.mkdir(dstSubDir)
            firstSubDirFlag = False
        else: # 第二级子目录
            subToothIndexDir = root.split("/")[-1]
            for f in fs:
                fName, fType = tuple(f.split("."))
                if fType == "obj":
                    objSrcFile = os.path.join(root, f)
                    offDstFile = os.path.join(dstRootDir, subToothIndexDir+"/"+fName+".off")
                    __obj2off(objSrcFile, offDstFile)

def convertNpy2Txt(srcNpyRoot, dstTxtRoot):
    if not os.path.exists(srcNpyRoot):
        print("Source directory does not exist.")
        return
    if not os.path.exists(dstTxtRoot):
        os.mkdir(dstTxtRoot)
    
    
    firstSubDirFlag = True
    for root, dirs, fs in os.walk(srcNpyRoot):
        if firstSubDirFlag: #第一级目录
            firstSubDirFlag = False
            dstSubDirs = [os.path.join(dstTxtDir,dir) for dir in dirs]
            for dstSubDir in dstSubDirs:
                if not os.path.exists(dstSubDir):
                    os.mkdir(dstSubDir)
            continue
        else: # 第二级子目录
            subToothIndexDir = root.split("/")[-1]
            for f in fs:
                fName, fType = tuple(f.split("."))
                if fType == "npy":
                    srcNpyFile = os.path.join(root, f)
                    dstTxtFile = os.path.join(dstTxtRoot, subToothIndexDir+"/"+fName+".txt")
                    tempArray = np.load(srcNpyFile)
                    np.savetxt(dstTxtFile, tempArray)
                    


if __name__ == "__main__":
    # convertObj2Off("./data/format-obj/", "./data/format-off/") # 路径以/结尾 #生成off格式的单颗牙齿数据
    srcNpyDir = "./data/repaired-npy/"
    dstTxtDir = "./data/repaired-txt/"
    convertNpy2Txt(srcNpyDir, dstTxtDir)
import numpy as np
import igl
from vispy.io import mesh as vismesh
import os
import pandas as pd
import json
import shutil
import glob
from pcd_mesh_utils import getLargestConnectedMeshComponent, removeMeshSmallComponents
from separate_raw import getLabels, getMeshVF, extractVerticesFaces, getTransformMat, transformVertexPos, clearPrevOutput


"""生成切牙数据中对应不同牙齿的txt顶点数组文件, 并且进行分牙数据处理"""
"""切牙数据中的口扫label.txt中的每个label与每个顶点相对应"""

REFINED_DATA_PATH = "./data-refined/data-refined/"
AUXILIARY_LABEL_PATH = "data-refined/labelAuxil/"
SAVE_REFINED_PATH = "./data-refined/refined-txt/"
DST_TXT_DIR = "./data-refined/format-txt/"
DST_OBJ_DIR = "./data-refined/format-obj/"
SCAN_OBJ_TAG = "cutMesh"
SCAN_TXT_TAG = "cutResult"
REFI_OBJ_TAG = "toothMesh"
AUXIL_TXT_TAG = "auxiliaryInfo"
ID_TAG = "tag"
COLUMN_NAMES = [ID_TAG, SCAN_OBJ_TAG, SCAN_TXT_TAG, REFI_OBJ_TAG, AUXIL_TXT_TAG]



def turnVertexLabelIntoFaceLabel(faces, vertexLabel):
    """facelabel/vertexLabel: 0,11-18,21-28,31-38,41-48"""
    """将三角面片的label定义为它第一个顶点的label"""
    faceVLabels = vertexLabel[faces]
    mask = np.all(np.vstack([faceVLabels[:,0]==faceVLabels[:,1], faceVLabels[:,0]==faceVLabels[:,2]]), axis=0)
    faceLabel = np.zeros((faces.shape[0],), dtype=faces.dtype)
    faceLabel[mask] = faceVLabels[mask,0]
    return faceLabel

def getToothIndexAndCenters(auxiliaryInfoTxt):
    """从附加的信息txt文件获取企业修复的切牙数据的每颗牙齿的id和重心位置"""
    with open(auxiliaryInfoTxt, "r") as f:
        teethInfo = json.loads(f.readlines()[0])
    toothIndexCentersDict = {}
    for toothInfo in teethInfo:
        toothIndexCentersDict[int(toothInfo["name"])] = np.array([float(toothInfo["glocalOrigin"]["x"]),\
            float(toothInfo["glocalOrigin"]["y"]),float(toothInfo["glocalOrigin"]["z"])], dtype=np.float32)
    # print(toothIndexCentersDict)
    return toothIndexCentersDict


def getTaggedFileInfoDF(srcDataDir, auxInfoDir):
    """将口扫obj、口扫Label.txt与切牙obj配对"""
    """切牙数据中的口扫label.txt中的每个label与每个顶点相对应"""
    allNames = set([os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(srcDataDir,"*.obj"))])
    allNames = list(allNames)
    allNames.sort()
    nameIndexMap = {name:index for index,name in enumerate(allNames)}
    
    indexObjTxtObjDicts = {}
    for f in os.listdir(srcDataDir):
        fName, fType = os.path.splitext(f)
        try:
            name, fDetail, UorL = tuple(fName.split("_"))
        except:
            print("Error in parsing file name")
        idkey = name + UorL.capitalize()
        if not (idkey in indexObjTxtObjDicts):
            indexObjTxtObjDicts[idkey] = {ID_TAG: str(nameIndexMap[name])+UorL.capitalize(), }
        assert fDetail in COLUMN_NAMES, "Inconsistent file name: {}".format(f)
        indexObjTxtObjDicts[idkey][fDetail] = f
    for f in os.listdir(auxInfoDir):
        fName, fType = os.path.splitext(f)
        try:
            splitFName = fName.split("_")
            name, UorL = splitFName[0], splitFName[-1]
        except:
            print("Error in parsing file name")
        idkey = name + UorL.capitalize()
        if idkey in indexObjTxtObjDicts:
            indexObjTxtObjDicts[idkey][AUXIL_TXT_TAG] = f

    df = pd.DataFrame(indexObjTxtObjDicts.values(), columns=COLUMN_NAMES)
    # df = df.dropna(axis=0,how='any')
    # df.to_csv("temp.csv")
    return df



def separateRefinedTeethVertices(refinedObj, auxiliaryLabelTxt):
    """对修补后的切牙数据进行分牙操作"""
    refiV, refiF = getMeshVF(refinedObj)
    refiF = refiF.astype(np.int32)
    # print("refined vertices shape: ", refiV.shape)
    # print("refined faces shape: ", refiF.shape)
    refiVLabels = igl.vertex_components(refiF) # 根据连通情况将顶点分组
    refiVGroups = [refiV[refiVLabels==lab] for lab in np.unique(refiVLabels)]
    refiVGroupCenter = np.array([np.mean(vv,axis=0) for vv in refiVGroups]) #补全的切牙数据每个牙齿的重心坐标
    taggedRefiVs = {}

    toothIndexCentersDict = getToothIndexAndCenters(auxiliaryLabelTxt)
    for toothID, toothCenter in toothIndexCentersDict.items():
        centerDist = np.linalg.norm(toothCenter-refiVGroupCenter, ord=2, axis=1) 
        refiVGrpCorreId= np.argmin(centerDist)
        taggedRefiVs[toothID] = refiVGroups[refiVGrpCorreId]
    return taggedRefiVs

def saveTeethPointsTxt(txtName, savePath, taggedRefiVs):
    """保存修补的切牙数据的每颗分牙 taggedRefiVs: Dict"""
    for i in taggedRefiVs.keys():
        tempPath = os.path.join(savePath,"{}/".format(i))
        if not os.path.exists(tempPath):
            os.makedirs(tempPath)

    for idTag, arr in taggedRefiVs.items():
        toothDir = os.path.join(savePath, "{}/".format(idTag))
        # # 获取当前特定牙齿ID路径下数据最大编号
        # indexList = [int(f.split(".")[0]) for f in os.listdir(toothDir)]
        # curFid = 0
        # if len(indexList) > 0:
        #     curFid = np.max(indexList) + 1
        fileName = os.path.join(toothDir, "{}.txt".format(txtName))
        np.savetxt(fileName, arr)



def separateScannedTeethVertices(idTag, scannedObj, scannedVertexLabelTxt, auxiliaryLabelTxt, txtOutputPath, objOutputPath):
    """对修补前的数据进行分牙操作"""
    vertexLabels = getLabels(scannedVertexLabelTxt) #顶点label
    print("labels shape: ", vertexLabels.shape)
    uniqueVertexLabels = np.unique(vertexLabels)
    for j in range(len(uniqueVertexLabels)): #确认uniqueVertexLabels满足从0开始的排列
        assert j == uniqueVertexLabels[j]
    print("unique labels:", uniqueVertexLabels)

    scanV, scanF = getMeshVF(scannedObj)
    print("scanned vertices shape: ", scanV.shape)
    print("scanned faces shape: ", scanF.shape)
    scanVGroups = [scanV[vertexLabels==lab] for lab in uniqueVertexLabels]
    scanVGroupCenter = np.array([np.mean(vv,axis=0) for vv in scanVGroups])
    vertexLabelMapping = np.zeros((len(uniqueVertexLabels),), dtype=np.int32) #0表示牙龈 1-15/16表示牙齿 需要映射到0,11-18,21-28,31-38,41-48
    
    taggedScannedVs = {}
    toothIndexCentersDict = getToothIndexAndCenters(auxiliaryLabelTxt)
    for toothID, toothCenter in toothIndexCentersDict.items(): #通过重心位置判断对应牙齿编号
        centerDist = np.linalg.norm(toothCenter-scanVGroupCenter, ord=2, axis=1) 
        scanVGrpCorreId = np.argmin(centerDist)
        taggedScannedVs[toothID] = scanVGroups[scanVGrpCorreId]
        vertexLabelMapping[scanVGrpCorreId] = toothID
     
    # 重新修改vertexLabel，修改为0,11-18,21-28,31-38,41-48
    vertexLabels = vertexLabelMapping[vertexLabels]
    faceLabels = turnVertexLabelIntoFaceLabel(scanF, vertexLabels)

    """生成一个样本(上牙列或下压列)每颗牙齿的obj文件和txt文件,每个牙列经过平移旋转等位置调整"""
    transMat = getTransformMat(taggedScannedVs)
    vertices = transformVertexPos(scanV, transMat)
    
    uniqueLabels = np.unique(faceLabels)
    for toothLabelID in uniqueLabels:
        if toothLabelID == 0:
            continue
        toothMask = faceLabels==toothLabelID
        toothVertices, toothFaces = extractVerticesFaces(toothMask, vertices, scanF) #获得对应牙齿的顶点和三角面片，顶点重新标号
        # toothVertices, toothFaces, goodConnectionFlag = getLargestConnectedMeshComponent(toothVertices, toothFaces) #获取最大连接区域(顶点数量最多)的mesh
        # if not goodConnectionFlag:
        #     print("bad connection! skip tag: {}/{}".format(toothLabelID,idTag))
        #     continue
        # toothVertices, toothFaces = removeMeshSmallComponents(toothVertices, toothFaces, minNumVertex2Keep=500)
        

        txtOutputDir = os.path.join(txtOutputPath, "{}/".format(toothLabelID))
        # curIndex = 0
        # # 获取当前特定牙齿ID路径下数据最大编号
        # indexList = [int(f.split(".")[0]) for f in os.listdir(txtOutputDir)]
        # if len(indexList) > 0:
        #     curIndex = np.max(indexList) + 1
        txtOutputFile = os.path.join(txtOutputDir, "{}.txt".format(idTag))
        np.savetxt(txtOutputFile, toothVertices)

        objOutputDir = os.path.join(objOutputPath, "{}/".format(toothLabelID))
        objOutputFile = os.path.join(objOutputDir, "{}.obj".format(idTag))
        vismesh.write_mesh(objOutputFile, toothVertices, toothFaces, normals=None, texcoords=None, overwrite=True)
        print("Write mesh: {}".format(objOutputFile))





if __name__ == "__main__":
    df = getTaggedFileInfoDF(REFINED_DATA_PATH, AUXILIARY_LABEL_PATH)
    df.to_csv(r"./data-refined/nameIndexMapping.csv", index=False)
    print(df)

    df = df.dropna(axis=0,how='any') #删除存在空值的行；部分数据缺少对应标签

    # 删除之前保存的企业补全的切牙的点云txt
    if os.path.exists(SAVE_REFINED_PATH):
        shutil.rmtree(SAVE_REFINED_PATH)
    
    # 删除口扫分牙的obj，txt
    clearPrevOutput(DST_OBJ_DIR)
    clearPrevOutput(DST_TXT_DIR)

    for index, row in df.iterrows():
        scanObj = os.path.join(REFINED_DATA_PATH, row[SCAN_OBJ_TAG])
        scanVertexLabelTxt = os.path.join(REFINED_DATA_PATH, row[SCAN_TXT_TAG])
        refinedObj = os.path.join(REFINED_DATA_PATH, row[REFI_OBJ_TAG])
        auxilTxt = os.path.join(AUXILIARY_LABEL_PATH, row[AUXIL_TXT_TAG])
        tag = row[ID_TAG]

        # 对修补后的切牙数据进行分牙操作
        taggedRefiVs = separateRefinedTeethVertices(refinedObj, auxilTxt)
        saveTeethPointsTxt(tag, SAVE_REFINED_PATH, taggedRefiVs)

        # 对原始数据进行分牙操作
        separateScannedTeethVertices(tag, scanObj, scanVertexLabelTxt, auxilTxt, DST_TXT_DIR, DST_OBJ_DIR)
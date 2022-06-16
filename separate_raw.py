import numpy as np
import os
import shutil
import pandas as pd
from vispy.io import mesh as vismesh
import ray
import psutil
import glob


"""生成原始数据分牙的txt和obj格式的数据，单颗牙齿不封闭，每颗牙齿是一个曲面，需要fix_mesh.py进行修复"""
"""原始数据中的label.txt中的每个label与每个三角面相对应"""

SRC_DIR = r"./data/allData86Sample/"
DST_TXT_DIR = r"./data/format-txt/"
DST_OBJ_DIR = r"./data/format-obj/"



def extractVerticesFaces(faceLabelCondition, vertices, faces):
    """ 根据faces的labels条件，选择对应的节点，三角面片
    """
    selectedFaces = faces[faceLabelCondition] 
    selectedVerticesIndex = pd.unique(selectedFaces.flatten()) # pandas unique不排序 numpy unique 默认排序
    vertexIndexMapping = np.zeros((np.max(selectedVerticesIndex)+1,), dtype=np.uint32) #重新映射vertex index
    vertexIndexMapping[selectedVerticesIndex] = np.arange(selectedVerticesIndex.shape[0])
    selectedVertices = vertices[selectedVerticesIndex]
    selectedFaces = vertexIndexMapping[selectedFaces] #重新设置三角面片顶点index, [统一]顶点出现顺序：顶点index需满足从小到大排列
    return selectedVertices, selectedFaces


def getTransformMat(taggedVGroups):
    # 上牙列：tag全部小于UL_THRE，下压列：tag全部大于UL_THRE，利用牙列的位置关系进行粗对准
    tags = np.array(list(taggedVGroups.keys()), dtype=np.uint8)
    
    isLowerTeeth = (np.max(tags)>=30)
    LR_SEPA = 40 if isLowerTeeth else 20

    lastRightTag = np.max(tags[tags<LR_SEPA]) if isLowerTeeth else np.max(tags)
    lastLeftTag = np.max(tags) if isLowerTeeth else np.max(tags[tags<LR_SEPA])
    firstLeftTag = np.min(tags[tags<LR_SEPA])
    firstRightTag = np.min(tags[tags>=LR_SEPA])

    lastLeftCenter = np.mean(taggedVGroups[lastLeftTag], axis=0)
    lastRightCenter = np.mean(taggedVGroups[lastRightTag], axis=0)
    translatedOrigin = (lastLeftCenter+lastRightCenter)/2.
    translatedMat = np.identity(4, dtype=np.float32) #平移矩阵
    translatedMat[3,0:3] = -translatedOrigin

    firstLeftCenter = np.mean(taggedVGroups[firstLeftTag], axis=0)
    firstRightCenter = np.mean(taggedVGroups[firstRightTag], axis=0)
    rotated_OZ = (firstLeftCenter+firstRightCenter)/2. - translatedOrigin
    rotated_approx_OX = lastRightCenter - lastLeftCenter
    rotated_OY = np.cross(rotated_OZ, rotated_approx_OX)

    vec_OZ = np.array([0., 0., 1.], dtype=np.float32)
    unit_rot_OZ = rotated_OZ / np.linalg.norm(rotated_OZ, ord=2)
    angleCosVal = np.dot(vec_OZ, unit_rot_OZ)
    angle = np.arccos(angleCosVal)

    rotAxis = np.cross(unit_rot_OZ, vec_OZ)
    rotAxis = rotAxis / np.linalg.norm(rotAxis, ord=2)
    K = np.array([[0., -rotAxis[2], rotAxis[1]],\
        [rotAxis[2], 0., -rotAxis[0]],\
        [-rotAxis[1], rotAxis[0], 0.]], dtype=np.float32)

    LeftRotMat = np.identity(3) + np.sin(angle) * K + (1.-np.cos(angle)) * (K @ K)
    RotMat = np.identity(4, dtype=np.float32)
    RotMat[:3,:3] = LeftRotMat.T

    prevHomoTransOY = np.hstack([rotated_OY, [1.]]) @ RotMat
    unitPrevTransOY = prevHomoTransOY[:3] / np.linalg.norm(prevHomoTransOY[:3], ord=2)

    vec_OY = np.array([0., 1., 0.], dtype=np.float32)
    angle_ROZ = np.arccos(np.dot(vec_OY, unitPrevTransOY))
    
    if np.cross(unitPrevTransOY, vec_OY)[2]<0:
        angle_ROZ = -angle_ROZ
    rotOZMat = np.identity(4, dtype=np.float32)
    rotOZMat[:2,:2] = np.array([[np.cos(angle_ROZ),np.sin(angle_ROZ)],[-np.sin(angle_ROZ),np.cos(angle_ROZ)]], dtype=np.float32)

    transMat = translatedMat @ RotMat @ rotOZMat
    return transMat

    
def getLabels(labelTxt):
    delimiter = None
    with open(labelTxt, "r") as f:
        if "," in f.read(4): # 读取label文件前4个字符，确认分隔符
            delimiter = ","
    labels = np.loadtxt(labelTxt, delimiter=delimiter, dtype=np.uint8)
    return labels


def getMeshVF(meshObj):
    vertices, faces, normals, texcoords = vismesh.read_mesh(meshObj)
    return vertices, faces


def getTaggedVertexGroup(vertices, faces, faceLabels):
    uniqueLabels = np.unique(faceLabels)
    taggedVertexGroup = {}
    for toothLabelID in uniqueLabels:
        if toothLabelID == 0:
            continue
        toothMask = (faceLabels==toothLabelID)
        toothVertices, _ = extractVerticesFaces(toothMask, vertices, faces)
        taggedVertexGroup[toothLabelID] = toothVertices
    return taggedVertexGroup


def transformVertexPos(vertices, transMat):
    """vertices: (n,3), transMat: Matrix 4*4"""
    n = vertices.shape[0]
    homoV = np.hstack([vertices, np.ones(shape=(n,1))])
    transHomoV = homoV @ transMat
    return transHomoV[:,:3]



def getTransMeshVF(vertices, faces, faceLabels):
    taggedVertexGroup = getTaggedVertexGroup(vertices, faces, faceLabels)
    transMat = getTransformMat(taggedVertexGroup)
    transVertices = transformVertexPos(vertices, transMat)
    return transVertices, faces



def writeTransObj(srcMeshObj, faceLabelTxt, dstMeshObj):
    vertices, faces = getMeshVF(srcMeshObj)
    faceLabels = getLabels(faceLabelTxt)
    transVertices, faces = getTransMeshVF(vertices, faces, faceLabels)
    vismesh.write_mesh(dstMeshObj, transVertices, faces, normals=None, texcoords=None, overwrite=True)
    print("Write mesh: {}".format(dstMeshObj))



def printMeshAndLabelsStatistics(objMeshFile, labelTxtPath):
    """Test function: No output"""
    vertices, faces, normals, texcoords = vismesh.read_mesh(objMeshFile)
    labels = getLabels(labelTxtPath)
    uniqueLabels = np.unique(labels)
    #顶点读取顺序由faces中顶点index出现顺序决定，若faces中顶点index出现顺序满足由小到大排序，则等价于按行读取顶点
    print("obj file path: ", objMeshFile)
    print("labels shape: ", labels.shape)
    print("unique labels:", uniqueLabels) # 三角面片label 0表示牙龈。上颌牙齿使用1或2开头的编号，下颌用3或4开头的编号。
    print("vertices dtype: ", vertices.dtype)
    print("vertices shape: ", vertices.shape)
    print("Min index of vertices: ", np.min(faces))
    print("Max index of vertices: ", np.max(faces))
    print("faces dtype: ", faces.dtype)
    print("faces shape: ", faces.shape)
    print("normals dtype: ", normals.dtype)
    print("normals shape: ", normals.shape)
    print("texcoords: ",texcoords)



@ray.remote
def generateSeparatedToothData(srcMeshObj, faceLabelTxt, txtOutputPath, objOutputPath, fileBaseName):
    """生成一个样本（上牙列或下压列）每颗牙齿的obj文件和txt文件，每个牙列经过平移旋转等位置调整"""

    vertices, faces = getMeshVF(srcMeshObj)
    faceLabels = getLabels(faceLabelTxt)
    vertices, faces = getTransMeshVF(vertices, faces, faceLabels)
    
    uniqueLabels = np.unique(faceLabels)
    for toothLabelID in uniqueLabels:
        if toothLabelID == 0:
            continue
        toothMask = faceLabels==toothLabelID
        toothVertices, toothFaces = extractVerticesFaces(toothMask, vertices, faces)
        
        txtOutputDir = os.path.join(txtOutputPath, "{}/".format(toothLabelID))

        txtOutputFile = os.path.join(txtOutputDir, "{}.txt".format(fileBaseName))
        np.savetxt(txtOutputFile, toothVertices)

        objOutputDir = os.path.join(objOutputPath, "{}/".format(toothLabelID))
        objOutputFile = os.path.join(objOutputDir, "{}.obj".format(fileBaseName))
        vismesh.write_mesh(objOutputFile, toothVertices, toothFaces, normals=None, texcoords=None, overwrite=True)
        print("Write mesh: {}".format(objOutputFile))




def gettaggedSrcObjTxtList(srcRootDir):
    if not os.path.exists(srcRootDir):
        print("Put the original obj data and txt label into the directory {}".format(srcRootDir))
        return

    allNames = set([os.path.basename(f).split("_")[0] for f in glob.glob(os.path.join(srcRootDir,"*.obj"))])
    allNames = list(allNames)
    allNames.sort()
    nameIndexMap = {name:index for index,name in enumerate(allNames)}
    taggedObjTxt = [] # 获取obj数据和txt label文件名的配对
    for f in os.listdir(srcRootDir):
        fName, fType = os.path.splitext(f)
        name, ul = tuple(fName.split("_")) # example: baianan14_l.obj
        tag = str(nameIndexMap[name])+ul.upper()
        if fType == ".obj":
            taggedObjTxt.append( (tag, os.path.join(srcRootDir,f), os.path.join(srcRootDir,fName+".txt")) )
    return taggedObjTxt


def clearPrevOutput(dstPath):
    """清除上次输出，重建目录结构"""
    if os.path.exists(dstPath):
        shutil.rmtree(dstPath)
    if not os.path.exists(dstPath):
        os.makedirs(dstPath)
    toothIndexDir = [i*10+j for i in range(1,5) for j in range(1,9)]
    for id in toothIndexDir:
        os.makedirs(os.path.join(dstPath, str(id)))



def testTransObj():
    """测试上下牙列的位置变换"""
    srcMeshObj = r"data/allData86Sample/baianan14_l.obj"
    faceLabelTxt = r"data/allData86Sample/baianan14_l.txt"
    dstMeshObj = r"test.obj"
    writeTransObj(srcMeshObj, faceLabelTxt, dstMeshObj)




if __name__ == "__main__":
    # clearPrevOutput(DST_OBJ_DIR)
    # clearPrevOutput(DST_TXT_DIR)
    taggedObjTxt = gettaggedSrcObjTxtList(SRC_DIR)
    df = pd.DataFrame(taggedObjTxt, columns= ['tag','srcObj','srcTxt'])
    df.to_csv(r"./data/nameIndexMapping.csv", index=False)


    # # Single thread
    # for (fBsName, objF,txtF) in taggedObjTxt:
    #     generateSeparatedToothData(objF, txtF, DST_TXT_DIR, DST_OBJ_DIR, fBsName)
    
    num_cpus = psutil.cpu_count(logical=False)
    ray.init(num_cpus=num_cpus)
    ray.get([generateSeparatedToothData.remote(objF, txtF, DST_TXT_DIR, DST_OBJ_DIR, fBsName) for (fBsName,objF,txtF) in taggedObjTxt])


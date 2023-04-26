import functools
import itertools
import os
from collections import Counter
from copy import copy, deepcopy

import cycpd
import h5py
import numpy as np
import open3d as o3d
import ray
import scipy
import scipy.io
import skimage
from matplotlib import pyplot as plt
from scipy.spatial import Delaunay, distance_matrix
from scipy.spatial.transform import Rotation as RR
from shapely.geometry import MultiLineString, Point
from shapely.ops import polygonize, unary_union

from const import PHOTO

print = functools.partial(print, flush=True)


class EMOpt5Views(object):
    def __init__(
        self,
        edgeMasks,
        photoTypes,
        visMasks,
        Mask,
        Mu,
        Mu_normals,
        SqrtEigVals,
        Sigma,
        PoseCovMats,
        ScaleCovMat,
        transVecStd,
        rotVecStd,
    ) -> None:
        self.photoTypes = sorted(photoTypes, key=lambda x: x.value)
        assert self.photoTypes == [
            PHOTO.UPPER,
            PHOTO.LOWER,
            PHOTO.LEFT,
            PHOTO.RIGHT,
            PHOTO.FRONTAL,
        ]

        self.edgeMask = [None] * 5  # order as the Enum value in PHOTO
        self.visIdx = [None] * 5
        for phType, eMask, visMask in zip(photoTypes, edgeMasks, visMasks):
            assert eMask.ndim == 2, "edgeMask should be grayscale"  #
            self.edgeMask[phType.value] = eMask  # binary 2d-array
            self.visIdx[phType.value] = np.argwhere(visMask[Mask] > 0).flatten()

        self.P_true = [
            np.argwhere(v > 0)[:, ::-1] for v in self.edgeMask
        ]  # xy position of detected boundary points in image coordinates
        self.P_true_normals = [
            self.initEdgeMaskNormals(v) for v in self.P_true
        ]  # normals at the detected boundary points in image coordinates
        self.M = [
            len(v) for v in self.P_true
        ]  # nums of detected boundary points in different photo
        self.flag_99_percentile = [np.ones((m,), np.bool_) for m in self.M]
        self.P_true_99_percentile = [
            p_true[flag] for flag, p_true in zip(self.flag_99_percentile, self.P_true)
        ]

        self.Mask = Mask
        Mask_U, Mask_L = np.split(Mask, 2, axis=0)
        self.numUpperTooth = int(np.sum(Mask_U))
        self.numTooth = int(np.sum(Mask))
        self.numPoint = Mu.shape[1]
        self.ul_sp = {
            phType.value: np.argwhere(
                self.visIdx[phType.value] >= self.numUpperTooth
            ).min()
            for phType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
        }

        self.X_Mu = Mu[Mask]
        self.X_Mu_normals = Mu_normals[Mask]
        self.X_Mu_centroids = self.X_Mu.mean(axis=1)

        self.SqrtEigVals = SqrtEigVals[Mask]  # shape=(self.numTooth, 1, numPC)
        self.SigmaT = np.transpose(Sigma[Mask], (0, 2, 1))

        self.meanRotVecXYZs = np.zeros((self.numTooth, 3))
        self.meanTransVecXYZs = np.zeros((self.numTooth, 3))
        self.meanScales = np.ones((self.numTooth,))
        self.invCovMats = np.linalg.inv(PoseCovMats[Mask])
        self.invCovMatOfScale = np.linalg.inv(ScaleCovMat[Mask][:, Mask])

        # init teeth shape subspace
        self.numPC = SqrtEigVals.shape[-1]
        self.featureVec = np.zeros(
            self.SqrtEigVals.shape, dtype=np.float32
        )  # shape=(self.numTooth, 1, numPC), mean=0, std=1

        # init teeth scales, rotation vecs around X-Y-Z axes, translation vectors along X-Y-Z axes
        self.scales = np.ones((self.numTooth,), np.float32)
        self.rotVecXYZs = np.zeros((self.numTooth, 3), np.float32)
        self.transVecXYZs = np.zeros((self.numTooth, 3), np.float32)

        self.rowScaleXZ = np.ones(
            (2,), dtype=np.float32
        )  # anistropic tooth row scale in xz plane, optimized in stage1, converted into tooth scales and transVecXYZs before stage 2

        # init extrinsic param of camera
        self.ex_rxyz_default = {
            PHOTO.UPPER: np.array([0.7 * np.pi, 0.0, 0.0], dtype=np.float32),  # upper
            PHOTO.LOWER: np.array([-0.7 * np.pi, 0.0, 0.0], dtype=np.float32),  # lower
            PHOTO.LEFT: np.array([2.99, 0, -0.97], dtype=np.float32),  # left
            PHOTO.RIGHT: np.array([2.99, 0, 0.97], dtype=np.float32),  # right
            PHOTO.FRONTAL: np.array([np.pi, 0.0, 0.0], dtype=np.float32),
        }  # frontal
        self.ex_txyz_default = {
            PHOTO.UPPER: np.array([0.0, 0.0, 70.0], dtype=np.float32),  # upper # 70
            PHOTO.LOWER: np.array([0.0, 0.0, 70.0], dtype=np.float32),  # lower # 70
            PHOTO.LEFT: np.array(
                [-5.0, 0.0, 120.0], dtype=np.float32
            ),  # left # [-5,0,70]
            PHOTO.RIGHT: np.array(
                [5.0, 0.0, 120.0], dtype=np.float32
            ),  # right # [5,0,70]
            PHOTO.FRONTAL: np.array([0.0, -2.0, 120.0], dtype=np.float32),
        }  # frontal # [0,-2,70]
        self.ex_rxyz = np.empty(
            (5, 3), dtype=np.float32
        )  # shape=(5,3) # init rot vecs around x-y-z axis based on photoType
        self.ex_txyz = np.empty(
            (5, 3), dtype=np.float32
        )  # shape=(5,3) # init trans vector
        # init intrinsic param of camera
        self.focLth = np.empty((5,), dtype=np.float32)
        self.dpix = np.empty((5,), dtype=np.float32)
        self.u0 = np.empty((5,), dtype=np.float32)
        self.v0 = np.empty((5,), dtype=np.float32)
        for photoType in self.photoTypes:
            self.initExtrIntrParams(photoType)

        # init relative pose of the lower tooth row with respect to the upper one
        self.rela_rxyz_default = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.rela_txyz_default = np.array([0.0, -5.0, 0.0], dtype=np.float32)
        self.rela_rxyz = None
        self.rela_R = None
        self.rela_txyz = None
        self.initRelativeToothRowPose()

        # approx. learning step (Not used)
        self.ex_rxyz_lr = 1.0
        self.ex_txyz_lr = 1.0
        self.focLth_lr = 1.0
        self.uv_lr = 1.0
        self.dpix_lr = 1.0
        self.rela_rxyz_lr = 1.0  # 0.001
        self.rela_txyz_lr = 1.0  # 0.1

        self.varAngle = 0.09  # param in expectation loss
        self.weight_point2point = (
            0.04  # param in residual pixel error in maximization loss
        )
        self.weight_point2plane = (
            2.0  # param in residual pixel error in maximization loss
        )
        # weight in maximization step for 5 views: [PHOTO.UPPER, PHOTO.LOWER, PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]
        self.weightViews = np.array(
            [1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32
        )  # [3,3,1,1,1]
        self.weightAniScale = 1.0
        self.weightTeethPose = (
            1.0  # param in residual teeth pose error in maximization loss
        )
        self.weightFeatureVec = (
            1.0  # param in residual featureVec error in maximization loss
        )

        self.transVecStd = transVecStd
        self.scaleStd = np.mean(np.sqrt(np.diag(ScaleCovMat)))
        self.rotVecStd = rotVecStd

        self.X_deformed = self.X_Mu.copy()
        self.X_deformed_normals = self.X_Mu_normals.copy()
        self.RotMats = np.tile(np.eye(3), (self.numTooth, 1, 1))
        self.X_trans = self.X_Mu.copy()
        self.X_trans_normals = self.X_Mu_normals.copy()

        self.extrViewMat = np.empty(
            (5, 4, 3), dtype=np.float32
        )  # homo world coord (xw,yw,zw,1) to camera coord (xc,yc,zc): 4*3 right-multiplying matrix
        self.X_camera = [
            None
        ] * 5  # compute X in camera coord based on X in world coord, ndarray, shape=(numTooth,1500,3)
        self.X_camera_normals = [None] * 5

        self.intrProjMat = np.empty(
            (5, 3, 3), dtype=np.float32
        )  # camera coord (xc,yc,zc) to image coord (u,v,zc): 3*3 right-multiplying matrix
        self.X_uv = [
            None
        ] * 5  # compute X in image coord based on X_camera in camera coord, ndarray, shape=(numTooth,1500,2)
        self.X_uv_normals = [None] * 5

        self.vis_hull_vertices = [None] * 5
        self.vis_hull_vertex_indices = [
            None
        ] * 5  # visible points in image coord, and corre idx in X
        self.P_pred = [None] * 5  # edgeMask prediction 2d-array, shape=(?,2)
        self.P_pred_normals = [None] * 5
        self.X_Mu_pred = [None] * 5  # correponding 3D points of P_pred in X_Mu
        self.X_Mu_pred_normals = [
            None
        ] * 5  # correponding 3D point normals of P_pred_normals in X_Mu_normals
        self.X_deformed_pred = [None] * 5
        self.X_deformed_pred_normals = [None] * 5
        self.SigmaT_segs = [None] * 5

        for phType in self.photoTypes:
            self.updateEdgePrediction(phType)

        self.loss_expectation_step = np.zeros((5,))
        self.corre_pred_idx = [None] * 5
        self.loss_maximization_step = 0.0

    @staticmethod
    def computePointNormals(X):
        # X.shape=(self.numTooth,self.numPoint,3)
        normals = []
        for vertices in X:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            pcd.estimate_normals()
            # to obtain a consistent normal orientation
            pcd.orient_normals_consistent_tangent_plane(k=30)
            pcd.normalize_normals()
            normals.append(np.asarray(pcd.normals, dtype=np.float32))
        return np.array(normals, dtype=np.float32)

    ###########################################
    ######### Initialization functions ########
    ###########################################

    def initEdgeMaskNormals(self, vertices_xy, show=False):
        M = len(vertices_xy)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(
            np.hstack([vertices_xy, 20 * np.random.rand(M, 1)])
        )
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(k=30)
        normals_xy = np.asarray(pcd.normals)[:, :2]
        pcd.normals = o3d.utility.Vector3dVector(
            np.hstack([normals_xy, np.zeros((M, 1))])
        )
        pcd.normalize_normals()
        if show == True:
            o3d.visualization.draw_geometries(
                [pcd],
                window_name="image edge normals estimation",
                width=800,
                height=600,
                left=50,
                top=50,
                point_show_normal=True,
            )
        return np.asarray(pcd.normals, dtype=np.float32)[:, :2]

    def initExtrinsicRotVecs(self, photoType):
        ph = photoType.value
        self.ex_rxyz[ph] = self.ex_rxyz_default[photoType].copy()

    def initExtrinsicTransVec(self, photoType):
        ph = photoType.value
        self.ex_txyz[ph] = self.ex_txyz_default[photoType].copy()

    def initCameraIntrinsicParams(self, photoType):
        ph = photoType.value
        focLth = {
            PHOTO.UPPER: 100.0,
            PHOTO.LOWER: 100.0,
            PHOTO.LEFT: 100.0,
            PHOTO.RIGHT: 100.0,
            PHOTO.FRONTAL: 100.0,
        }
        dpix = {
            PHOTO.UPPER: 0.1,
            PHOTO.LOWER: 0.1,
            PHOTO.LEFT: 0.06,
            PHOTO.RIGHT: 0.06,
            PHOTO.FRONTAL: 0.06,
        }
        self.focLth[ph] = focLth[photoType]
        self.dpix[ph] = dpix[photoType]
        self.u0[ph] = self.edgeMask[ph].shape[1] / 2.0  # img.width/2
        self.v0[ph] = self.edgeMask[ph].shape[0] / 2.0  # img.height/2

    def initExtrIntrParams(self, photoType):
        self.initExtrinsicRotVecs(photoType)
        self.initExtrinsicTransVec(photoType)
        self.initCameraIntrinsicParams(photoType)

    def initRelativeToothRowPose(self):
        self.rela_rxyz = self.rela_rxyz_default.copy()
        self.rela_R = self.updateRelaRotMat(self.rela_rxyz)
        self.rela_txyz = self.rela_txyz_default.copy()

    @staticmethod
    def solveCameraParams(p2d, p3d):  # Direct Linear Transform to solve camera pose
        assert len(p2d) == len(p3d), "Nums of 2D/3D points should be equal."
        N = len(p2d)
        pxl_x, pxl_y = np.split(p2d, indices_or_sections=2, axis=1)
        X = np.hstack([p3d, np.ones((N, 1), np.float32)])  # shape=(N,4)
        O_Nx4 = np.zeros((N, 4), np.float32)
        A = np.vstack(
            [np.hstack([X, O_Nx4, -pxl_x * X]), np.hstack([O_Nx4, X, -pxl_y * X])]
        )
        u, s, vh = np.linalg.svd(A, full_matrices=True)
        p_sol = vh[np.argmin(s), :]
        P = p_sol.reshape(3, 4)
        _u, _s, _vh = np.linalg.svd(P, full_matrices=True)
        _c = _vh[-1, :]
        c = _c[:-1] / _c[-1]
        M = P[:, :3]
        Mt = P[:, 3]
        R, Q = scipy.linalg.rq(M)
        t = -Q @ c
        F = R / R[-1, -1]
        assert np.all(np.diag(F) > 0), "The diagonal values of R should be positive."
        assert np.allclose(R @ Q, M), "RQ Decomposition Failed."
        return Q, t, F  # return left-multiplying matrix

    @staticmethod
    @ray.remote
    def rigid_registration_2D(P_true, P_pred):
        X = P_true.astype(np.double)
        Y = P_pred.astype(np.double)
        reg = cycpd.rigid_registration(
            **{
                "X": X,
                "Y": Y,
                "max_iterations": 100,
                "tolerance": 1.0,
                "w": 1e-3,
                "verbose": False,
                "print_reg_params": False,
            }
        )
        TY, (s, r, t) = reg.register()
        return TY

    def updateCameraParams(
        self, p2d, p3d_lst, phType, rela_txyz, rela_R=np.identity(3)
    ):
        ph = phType.value
        if phType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            ul_sp = self.ul_sp[ph]
            p3d_lst = p3d_lst[:ul_sp] + [
                x @ rela_R + rela_txyz for x in p3d_lst[ul_sp:]
            ]
        _exRotMat, ex_t, _intrProjMat_T = self.solveCameraParams(
            p2d, np.vstack(p3d_lst)
        )
        self.ex_txyz[ph] = ex_t
        self.ex_rxyz[ph] = RR.from_matrix(_exRotMat).as_rotvec()
        self.focLth[ph] = (
            self.dpix[ph] * (_intrProjMat_T[0, 0] + _intrProjMat_T[1, 1]) / 2.0
        )
        self.u0[ph] = _intrProjMat_T[0, 2]
        self.v0[ph] = _intrProjMat_T[1, 2]
        print("Estimate camera params of ", phType)

    def assignValue2ExtrParamByName(
        self, photoType, paramName, value, assign2DefaultValue=False
    ):
        # print(paramName, value)
        # param name should be in ['r.x', 'r.y', 'r.z', 't.x', 't.y', 't.z']
        ph = photoType.value
        xyz2i = {"x": 0, "y": 1, "z": 2, "xyz": [0, 1, 2]}
        r_t, x_y_z = paramName.split(".")
        i = xyz2i[x_y_z]  # i=0,1,2
        if r_t == "r":
            self.ex_rxyz[ph, i] = value  # variable
            if assign2DefaultValue == True:
                self.ex_rxyz_default[photoType][i] = value
        elif r_t == "t":
            self.ex_txyz[ph, i] = value  # variable
            if assign2DefaultValue == True:
                self.ex_txyz_default[photoType][i] = value
        else:
            print(
                "param name should be in ['r.x', 'r.y', 'r.z', 'r.xyz', 't.x', 't.y', 't.z']"
            )

    def gridSearchExtrinsicParams(self):
        # search appropriate values for ex_rxyz
        ExtrParamSearchSpace = {
            PHOTO.UPPER: {
                "r.x": np.pi * np.array([0.6, 0.65, 0.7, 0.75, 0.8], np.float32)
            },
            PHOTO.LOWER: {
                "r.x": np.pi * np.array([-0.6, -0.65, -0.7, -0.75, -0.8], np.float32)
            },
            PHOTO.LEFT: {
                "r.xyz": np.array(
                    [
                        [3.11, 0, -0.49],
                        [3.05, 0, -0.73],
                        [2.99, 0, -0.97],
                        [2.90, 0, -1.20],
                        [2.80, 0, -1.43],
                    ],
                    np.float32,
                )
            },
            PHOTO.RIGHT: {
                "r.xyz": np.array(
                    [
                        [3.11, 0, 0.49],
                        [3.05, 0, 0.73],
                        [2.99, 0, 0.97],
                        [2.90, 0, 1.20],
                        [2.80, 0, 1.43],
                    ],
                    np.float32,
                )
            },
            PHOTO.FRONTAL: {"r.x": np.pi * np.array([0.98, 1.0, 1.02], np.float32)},
        }
        self.initRelativeToothRowPose()
        for phType, paramDict in ExtrParamSearchSpace.items():
            ph = phType.value
            for paramName, paramSearchSpace in paramDict.items():
                print(phType, paramName, paramSearchSpace)
                P_pred_list = []

                for paramValue in paramSearchSpace:
                    self.initExtrIntrParams(
                        phType
                    )  # init extrinsic and intrinsic camera params with default values
                    self.assignValue2ExtrParamByName(phType, paramName, paramValue)
                    self.updateEdgePrediction(phType)
                    P_pred_list.append(self.P_pred[ph])
                TY_list = ray.get(
                    [
                        self.rigid_registration_2D.remote(self.P_true[ph], _P_pred)
                        for _P_pred in P_pred_list
                    ]
                )

                losses = []
                for idx, paramValue in enumerate(paramSearchSpace):
                    self.initExtrIntrParams(
                        phType
                    )  # init extrinsic and intrinsic camera params with default values
                    self.assignValue2ExtrParamByName(phType, paramName, paramValue)
                    self.updateEdgePrediction(phType)
                    self.updateCameraParams(
                        TY_list[idx],
                        self.X_Mu_pred[ph],
                        phType,
                        self.rela_txyz,
                        self.rela_R,
                    )  # update extrinsic and intrinsic camera params
                    losses.append(
                        self.expectation_step(
                            0, phType, verbose=True, use_percentile=False
                        )
                    )  # use expectation loss as evaluation metric for extrinsic params

                idx_selected = np.argmin(losses)
                bestParamValue = paramSearchSpace[
                    idx_selected
                ]  # best guess from expectation loss
                print("Best param guess: ", bestParamValue)
                self.assignValue2ExtrParamByName(
                    phType, paramName, bestParamValue, assign2DefaultValue=True
                )  # update default values with the best guess
                self.initExtrIntrParams(
                    phType
                )  # init extrinsic and intrinsic camera params with default values
                self.updateEdgePrediction(phType)
                self.updateCameraParams(
                    TY_list[idx_selected],
                    self.X_Mu_pred[ph],
                    phType,
                    self.rela_txyz,
                    self.rela_R,
                )  # update extrinsic and intrinsic camera params
            print("-" * 50)

    def assignValue2RelaPoseParamByName(
        self, paramName, value, assign2DefaultValue=False
    ):
        # param name should be in ['rela.r.x', 'rela.r.y', 'rela.r.z', 'rela.t.x', 'rela.t.y', 'rela.t.z']
        print(paramName, value)
        xyz2i = {"x": 0, "y": 1, "z": 2}
        _, r_t, x_y_z = paramName.split(".")
        i = xyz2i[x_y_z]  # i=0,1,2
        if r_t == "r":
            self.rela_rxyz[i] = value  # variable
            if assign2DefaultValue == True:
                self.rela_rxyz_default[i] = value
        elif r_t == "t":
            self.rela_txyz[i] = value  # variable
            if assign2DefaultValue == True:
                self.rela_txyz_default[i] = value
        else:
            print(
                "param name should be in ['rela.r.x', 'rela.r.y', 'rela.r.z', 'rela.t.x', 'rela.t.y', 'rela.t.z']"
            )

    def gridSearchRelativePoseParams(self):
        # search appropriate values for rela_rxyz, rela_txyz
        RelativePoseParamSearchSpace = {
            "rela.t.z": (
                self.rela_txyz_default[2] + np.array([-3, -2, -1, 0, 1, 2, 3]),
                [PHOTO.LEFT, PHOTO.RIGHT],
            ),
            "rela.t.y": (
                np.array([-7, -6, -5, -4]),
                [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL],
            ),
            "rela.t.x": (
                np.array([-1.0, 0.0, 1.0]),
                [
                    PHOTO.FRONTAL,
                ],
            ),
        }
        for paramName, (
            paramSearchSpace,
            phTypes,
        ) in RelativePoseParamSearchSpace.items():
            self.initRelativeToothRowPose()
            num_photo_relevant = len(phTypes)
            P_true_list = []
            P_pred_list = []
            rela_txyz_list = []

            for paramValue in paramSearchSpace:
                self.assignValue2RelaPoseParamByName(paramName, paramValue)
                for phType in phTypes:
                    ph = phType.value
                    self.initExtrIntrParams(
                        phType
                    )  # init extrinsic and intrinsic camera params with default values
                    self.updateEdgePrediction(phType)
                    P_true_list.append(self.P_true[ph])
                    P_pred_list.append(self.P_pred[ph])
                    rela_txyz_list.append(self.rela_txyz)
            TY_list = ray.get(
                [
                    self.rigid_registration_2D.remote(_P_true, _P_pred)
                    for _P_true, _P_pred in zip(P_true_list, P_pred_list)
                ]
            )

            losses = []
            for idx, paramValue in enumerate(paramSearchSpace):
                loss = 0
                self.assignValue2RelaPoseParamByName(paramName, paramValue)
                for jdx, phType in enumerate(phTypes):
                    ph = phType.value
                    self.initExtrIntrParams(
                        phType
                    )  # init extrinsic and intrinsic camera params with default values
                    self.updateEdgePrediction(phType)
                    i = idx * num_photo_relevant + jdx
                    self.updateCameraParams(
                        TY_list[i], self.X_Mu_pred[ph], phType, rela_txyz_list[i]
                    )  # update extrinsic and intrinsic camera params
                    loss = loss + self.expectation_step(
                        0, phType, verbose=True, use_percentile=False
                    )  # use expectation loss as evaluation metric
                losses.append(loss)

            idx_selected = np.argmin(losses)
            bestParamValue = paramSearchSpace[
                idx_selected
            ]  # best guess from expectation loss
            self.assignValue2RelaPoseParamByName(
                paramName, bestParamValue, assign2DefaultValue=True
            )  # update default values with the best guess
            print("Best param guess: ", bestParamValue)
            for jdx, phType in enumerate(phTypes):
                ph = phType.value
                self.initExtrIntrParams(
                    phType
                )  # init extrinsic and intrinsic camera params with default values
                self.updateEdgePrediction(phType)
                i = idx_selected * num_photo_relevant + jdx
                self.updateCameraParams(
                    TY_list[i], self.X_Mu_pred[ph], phType, rela_txyz_list[i]
                )  # update extrinsic and intrinsic camera params
            print("-" * 50)

    def searchDefaultRelativePoseParams(self):
        # choose a good initial value for rela_txyz (whether it is overbite or underbite)
        phType = PHOTO.FRONTAL
        ph = phType.value
        RelativePoseParamSearchSpace = {"rela.t.z": np.array([0, 3, 6])}
        for paramName, SearchSpace in RelativePoseParamSearchSpace.items():
            P_pred_list = []
            rela_txyz_list = []
            for paramValue in SearchSpace:
                self.initExtrIntrParams(
                    phType
                )  # init extrinsic and intrinsic camera params with default values
                self.assignValue2RelaPoseParamByName(paramName, paramValue)
                self.updateEdgePrediction(phType)
                P_pred_list.append(self.P_pred[ph])
                rela_txyz_list.append(self.rela_txyz)
            TY_list = ray.get(
                [
                    self.rigid_registration_2D.remote(self.P_true[ph], _P_pred)
                    for _P_pred in P_pred_list
                ]
            )

            losses = []
            for idx, paramValue in enumerate(SearchSpace):
                self.initExtrIntrParams(
                    phType
                )  # init extrinsic and intrinsic camera params with default values
                self.assignValue2RelaPoseParamByName(paramName, paramValue)
                self.updateEdgePrediction(phType)  # 更新 X_Mu_pred
                self.updateCameraParams(
                    TY_list[idx], self.X_Mu_pred[ph], phType, rela_txyz_list[idx]
                )  # update extrinsic and intrinsic camera params
                losses.append(
                    self.expectation_step(0, phType, verbose=True, use_percentile=False)
                )  # use expectation loss as evaluation metric

            idx_selected = np.argmin(losses)
            bestParamValue = SearchSpace[
                idx_selected
            ]  # best guess from expectation loss
            self.assignValue2RelaPoseParamByName(
                paramName, bestParamValue, assign2DefaultValue=True
            )  # update default values with the best guess
            print("Best param guess: ", bestParamValue)
            self.initExtrIntrParams(
                phType
            )  # init extrinsic and intrinsic camera params with default values
            self.updateEdgePrediction(phType)
            self.updateCameraParams(
                TY_list[idx_selected],
                self.X_Mu_pred[ph],
                phType,
                rela_txyz_list[idx_selected],
            )  # update extrinsic and intrinsic camera params
        print("-" * 50)

    ###############################################
    # Deformatin in shape subspace for each tooth #
    ###############################################

    def updateDeformedPointPos(self, featureVec, tIdx):
        deformField = np.matmul(
            featureVec * self.SqrtEigVals[tIdx], self.SigmaT[tIdx]
        )  # shape=(numTooth,1,3*self.numPoint)
        return self.X_Mu[tIdx] + deformField.reshape(
            self.X_Mu[tIdx].shape
        )  # X_deformed

    def updateDeformedPointNomrals(self):
        # Normals of the deformed pointcloud are updated once each iteration due to computation cost
        # Implemented in the Expectation step
        pass

    ########################################################
    # Isotropic scaled rigid transformation for each tooth #
    ########################################################
    def computeRotMats(self, rotVecXYZs):
        rotMats = RR.from_rotvec(rotVecXYZs).as_matrix()
        rotMats = np.transpose(rotMats, (0, 2, 1))  # 变为右乘
        return rotMats

    def updateTransformedPointPos(
        self, X_deformed, scales, rotMats, transVecXYZs, tIdx
    ):
        # X_trans = scales_inv * (X_deformed + transVecXYZs_inv) @ rotMats_inv
        # in CPD: X_aligned_deformed = scales_cpd * X_trans @ rotMats_cpd + transVecXYZs_cpd
        return (
            np.multiply(
                scales[:, None, None],
                np.matmul(X_deformed - self.X_Mu_centroids[tIdx, None, :], rotMats),
            )
            + transVecXYZs[:, None, :]
            + self.X_Mu_centroids[tIdx, None, :]
        )

    def updateTransformedPointNormals(self, X_deformed_normals, rotMats):
        return np.matmul(X_deformed_normals, rotMats)

    ####################################################################
    # Relative Pose of Lower Tooth Row with respect to Upper Tooth Row #
    ####################################################################

    def updateRelaRotMat(self, rela_rxyz):
        return RR.from_rotvec(rela_rxyz).as_matrix().T

    def updateLowerPointPosByRelaPose(self, X_lower, rela_R, rela_txyz):
        return np.matmul(X_lower, rela_R) + rela_txyz

    def updateLowerPointNormalsByRelaPose(self, X_lower_normals, rela_R):
        return np.matmul(X_lower_normals, rela_R)

    ###############################
    # world coord -> camera coord #
    ###############################
    def updateExtrinsicViewMatrix(
        self, ex_rxyz, ex_txyz
    ):  # world coord to camera coord
        R = RR.from_rotvec(ex_rxyz).as_matrix().T  # for right-multiplying matrix
        return np.vstack([R, ex_txyz])  # Matrix 4*3

    def updatePointPosInCameraCoord(self, X_world, extrViewMat):
        # get 3D point cloud in camera coord, return array shape (n,3) or (batch,n,3)
        X_homo = np.concatenate([X_world, np.ones((*X_world.shape[:-1], 1))], axis=-1)
        return np.matmul(X_homo, extrViewMat)

    def updatePointNormalsInCameraCoord(self, X_world_normals, extrViewRotMat):
        return np.matmul(X_world_normals, extrViewRotMat)

    ##############################
    # camera coord ->image coord #
    ##############################

    def updateIntrinsicProjectionMatrix(
        self, focLth, dpix, u0, v0
    ):  # camera cood to image coord
        return np.array(
            [[focLth / dpix, 0.0, 0.0], [0.0, focLth / dpix, 0.0], [u0, v0, 1.0]]
        )

    def updatePointPosInImageCoord(self, X_camera, intrProjMat):
        # get 3D point cloud in image coord, return array shape (n,2) or (batch,n,2)
        # assert (X_camera[...,2]>0).all(), "max violation: {:.2f}".format(np.abs(np.min(X_camera[...,2]))) # Z-value of points should be positive
        invalid_z_val = X_camera[..., 2] < 0
        X_camera[..., 2][invalid_z_val] = 0.0
        X_image = np.matmul((X_camera / X_camera[..., [2]]), intrProjMat)
        X_uv = X_image[..., :2]
        return np.around(X_uv).astype(np.int32)

    def updatePointNormalsInImageCoord(self, X_camera_normals):
        X_cam_normals_xy = X_camera_normals[..., :2]
        return X_cam_normals_xy / np.linalg.norm(
            X_cam_normals_xy, axis=-1, keepdims=True
        )

    ##################################################
    # Extract contour pixels in projected pointcloud #
    ##################################################

    def __getUniquePixels(self, X_uv_int):
        # merge points at the same position in image coord
        # X_uv_int: array shape (n,2)dtype np.int32
        # pixels: array (m,2), each element represents (u_x, v_y)
        pixels, unique_indices = np.unique(X_uv_int, axis=0, return_index=True)
        return pixels, unique_indices

    def __getConcaveHullEdgeVertexIndices(
        self, coords, alpha
    ):  # coords is a 2D numpy array (u_x,v_y)
        tri = Delaunay(coords, qhull_options="Qt Qc Qz Q12").simplices
        ia, ib, ic = (
            tri[:, 0],
            tri[:, 1],
            tri[:, 2],
        )  # indices of each of the triangles' points
        pa, pb, pc = (
            coords[ia],
            coords[ib],
            coords[ic],
        )  # coordinates of each of the triangles' points
        a = np.linalg.norm(pa - pb, ord=2, axis=1)
        b = np.linalg.norm(pb - pc, ord=2, axis=1)
        c = np.linalg.norm(pc - pa, ord=2, axis=1)
        s = (a + b + c) * 0.5  # Semi-perimeter of triangle
        area = np.sqrt(
            s * (s - a) * (s - b) * (s - c)
        )  # Area of triangle by Heron's formula
        filter = (
            a * b * c / (4.0 * area) < 1.0 / alpha
        )  # Radius Filter based on alpha value
        edges = tri[filter]
        edges = [
            tuple(sorted(combo))
            for e in edges
            for combo in itertools.combinations(e, 2)
        ]
        count = Counter(edges)  # count occurrences of each edge
        edge_indices = [e for e, c in count.items() if c == 1]
        return np.array(edge_indices)

    def __constructConcaveHull(
        self, coords, edge_indices
    ):  # coords is a 2D numpy array (u_x,v_y)
        edges = [(coords[e[0]], coords[e[1]]) for e in edge_indices]
        ml = MultiLineString(edges)
        poly = polygonize(ml)
        hull = unary_union(list(poly))
        return hull

    def extractVisibleEdgePointsByAvgDepth(self, photoType):
        # X_uv: shape=(numTooth,numPoint,2), dtype=np.int32
        ph = photoType.value
        avg_depth = self.X_camera[ph][..., 2].mean(
            axis=1
        )  # avg_depth: array shape (numTooth,)
        tooth_order = avg_depth.argsort()
        X_uv_sort = self.X_uv[ph][tooth_order]
        hulls = []
        vis_hull_vs = []
        vis_hull_vids = []
        for x_uv in X_uv_sort:
            pixels, pixel_xuv_map = self.__getUniquePixels(x_uv)
            edge_v_indices = self.__getConcaveHullEdgeVertexIndices(pixels, alpha=0.1)
            hull = self.__constructConcaveHull(pixels, edge_v_indices)
            uni_edge_v_indices = np.unique(edge_v_indices)
            hull_v = x_uv[pixel_xuv_map[uni_edge_v_indices]]
            flags = np.ones((len(hull_v),), dtype=np.bool_)
            for i, v in enumerate(hull_v):
                for exist_hull in hulls:
                    if exist_hull.contains(Point(v)):
                        flags[i] = False
                        break
            if flags.any() == True:
                hulls.append(hull)
                vis_hull_vs.append(hull_v[flags])
                vis_hull_vids.append(pixel_xuv_map[uni_edge_v_indices[flags]])
            else:  # when the contour of a tooth disappears totally
                vis_hull_vs.append(
                    np.array([], dtype=hull_v.dtype).reshape((0, 2))
                )  # empty point
                vis_hull_vids.append([])
        # sort in the init order
        vis_hull_vs = [
            x
            for _, x in sorted(zip(tooth_order, vis_hull_vs), key=lambda pair: pair[0])
        ]
        vis_hull_vids = [
            x
            for _, x in sorted(
                zip(tooth_order, vis_hull_vids), key=lambda pair: pair[0]
            )
        ]
        return vis_hull_vs, vis_hull_vids

    ###########################################
    ######### Update in E step ################
    ###########################################

    def updateAlignedPointCloudInWorldCoord(self, stage, tIdx):
        if stage >= 3:
            self.X_deformed[tIdx] = self.updateDeformedPointPos(
                self.featureVec[tIdx], tIdx
            )
            self.X_deformed_normals[tIdx] = self.computePointNormals(
                self.X_deformed[tIdx]
            )
        if stage >= 2:
            self.RotMats[tIdx] = self.computeRotMats(self.rotVecXYZs[tIdx])
            self.X_trans[tIdx] = self.updateTransformedPointPos(
                self.X_deformed[tIdx],
                self.scales[tIdx],
                self.RotMats[tIdx],
                self.transVecXYZs[tIdx],
                tIdx,
            )
            self.X_trans_normals[tIdx] = self.updateTransformedPointNormals(
                self.X_deformed_normals[tIdx], self.RotMats[tIdx]
            )
        if stage == 1:
            self.X_trans[tIdx] = (
                np.hstack([self.rowScaleXZ[0], 1.0, self.rowScaleXZ[1]])
                * self.X_trans[tIdx]
            )  # self.rowScaleXZ = [1.,1.,1.] after maximization stage 1

    def updateEdgePrediction(self, photoType):
        ph = photoType.value
        tIdx = self.visIdx[ph]
        X_trans = self.X_trans[tIdx]  # upper
        X_trans_normals = self.X_trans_normals[tIdx]

        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            ul_sp = self.ul_sp[ph]
            self.rela_R = self.updateRelaRotMat(self.rela_rxyz)
            X_trans = np.concatenate(
                [
                    X_trans[:ul_sp],
                    self.updateLowerPointPosByRelaPose(
                        X_trans[ul_sp:], self.rela_R, self.rela_txyz
                    ),
                ],
                axis=0,
            )  # left, right, frontal
            X_trans_normals = np.concatenate(
                [
                    X_trans_normals[:ul_sp],
                    self.updateLowerPointNormalsByRelaPose(
                        X_trans_normals[ul_sp:], self.rela_R
                    ),
                ],
                axis=0,
            )

        self.extrViewMat[ph] = self.updateExtrinsicViewMatrix(
            self.ex_rxyz[ph], self.ex_txyz[ph]
        )  # homo world coord (xw,yw,zw,1) to camera coord (xc,yc,zc): 4*3 right-multiplying matrix
        self.X_camera[ph] = self.updatePointPosInCameraCoord(
            X_trans, self.extrViewMat[ph]
        )  # compute X in camera coord based on X in world coord, ndarray, shape=(numTooth,1500,3)
        self.X_camera_normals[ph] = self.updatePointNormalsInCameraCoord(
            X_trans_normals, self.extrViewMat[ph, :3, :]
        )

        self.intrProjMat[ph] = self.updateIntrinsicProjectionMatrix(
            self.focLth[ph], self.dpix[ph], self.u0[ph], self.v0[ph]
        )  # camera coord (xc,yc,zc) to image coord (u,v,zc): 3*3 right-multiplying matrix
        self.X_uv[ph] = self.updatePointPosInImageCoord(
            self.X_camera[ph], self.intrProjMat[ph]
        )  # compute X in image coord based on X_camera in camera coord, ndarray, shape=(numTooth,1500,2)
        self.X_uv_normals[ph] = self.updatePointNormalsInImageCoord(
            self.X_camera_normals[ph]
        )

        (
            self.vis_hull_vertices[ph],
            self.vis_hull_vertex_indices[ph],
        ) = self.extractVisibleEdgePointsByAvgDepth(
            photoType
        )  # visible points in image coord, and corre idx in X
        self.P_pred[ph] = np.vstack(
            self.vis_hull_vertices[ph]
        )  # edgeMask prediction 2d-array, shape=(?,2)
        self.P_pred_normals[ph] = np.vstack(
            [
                x[vis_hull_vids]
                for x, vis_hull_vids in zip(
                    self.X_uv_normals[ph], self.vis_hull_vertex_indices[ph]
                )
            ]
        )  # edgeMask normals prediction 2d-array, shape=(?,2)

        self.X_Mu_pred[ph] = [
            x[vis_hull_vids]
            for x, vis_hull_vids in zip(
                self.X_Mu[tIdx], self.vis_hull_vertex_indices[ph]
            )
        ]  # points in world coord corre to edgeMask prediction
        self.X_Mu_pred_normals[ph] = [
            x[vis_hull_vids]
            for x, vis_hull_vids in zip(
                self.X_Mu_normals[tIdx], self.vis_hull_vertex_indices[ph]
            )
        ]
        self.X_deformed_pred[ph] = [
            x[vis_hull_vids]
            for x, vis_hull_vids in zip(
                self.X_deformed[tIdx], self.vis_hull_vertex_indices[ph]
            )
        ]
        self.X_deformed_pred_normals[ph] = [
            x[vis_hull_vids]
            for x, vis_hull_vids in zip(
                self.X_deformed_normals[tIdx], self.vis_hull_vertex_indices[ph]
            )
        ]

    ###########################################
    ######### Update & Expectation Step #######
    ###########################################

    def expectation(self, photoType, verbose, use_percentile=True):
        # get point correspondence between detected contour points and projected contour points
        ph = photoType.value
        point_loss_mat = (
            distance_matrix(self.P_true[ph], self.P_pred[ph], p=2, threshold=1e8) ** 2
        )
        normal_loss_mat = (
            -((self.P_true_normals[ph] @ self.P_pred_normals[ph].T) ** 2)
            / self.varAngle
        )
        loss_mat = point_loss_mat * np.exp(normal_loss_mat)  # weighted loss matrix
        _corre_pred_idx = np.argmin(loss_mat, axis=1)
        losses = loss_mat[np.arange(self.M[ph]), _corre_pred_idx]
        if use_percentile == True:
            # l1_point_loss_mat = distance_matrix(self.P_true[ph], self.P_pred[ph], p=1, threshold=int(1e8))
            # l1_point_losses = l1_point_loss_mat[np.arange(self.M[ph]), _corre_pred_idx]
            # self.flag_99_percentile[ph] = l1_point_losses < (2.5 * 1.4826 * np.median(l1_point_losses))
            # 99-percentile
            self.flag_99_percentile[ph] = losses < np.percentile(losses, 99.0)
            self.corre_pred_idx[ph] = _corre_pred_idx[
                self.flag_99_percentile[ph]
            ]  # mapping from [0,M] to [0,len(num_pred_point)]
            self.P_true_99_percentile[ph] = self.P_true[ph][self.flag_99_percentile[ph]]
            self.loss_expectation_step[ph] = np.sum(losses[self.flag_99_percentile[ph]])
        else:
            self.corre_pred_idx[ph] = _corre_pred_idx
            self.loss_expectation_step[ph] = np.sum(losses)
        if verbose == True:
            print(
                "{} - unique pred points: {} - E-step loss: {:.2f}".format(
                    str(photoType),
                    len(np.unique(self.corre_pred_idx[ph])),
                    self.loss_expectation_step[ph],
                )
            )

    def expectation_step(self, stage, photoType, verbose=True, use_percentile=True):
        ph = photoType.value
        self.updateAlignedPointCloudInWorldCoord(stage, tIdx=self.visIdx[ph])
        self.updateEdgePrediction(photoType)
        self.expectation(photoType, verbose, use_percentile)
        return self.loss_expectation_step[ph]

    def expectation_step_5Views(self, stage, verbose=True):
        tIdx = [i for i in range(self.numTooth)]
        self.updateAlignedPointCloudInWorldCoord(stage, tIdx)
        for photoType in self.photoTypes:
            self.updateEdgePrediction(photoType)
            self.expectation(photoType, verbose, use_percentile=True)

    def save_expectation_step_result(self, fileName: str):
        xOptDict = self.get_current_e_step_result()
        xOptDict.update(
            {
                "np_invCovMatOfPose": self.invCovMats,
                "np_invCovMatOfScale": self.invCovMatOfScale,
                "np_X_Mu": self.X_Mu,
                "np_X_Mu_pred": self.X_Mu_pred,
                "np_X_Mu_pred_normals": self.X_Mu_pred_normals,
                "np_visIdx": self.visIdx,
                "np_corre_pred_idx": self.corre_pred_idx,
                "np_P_true": self.P_true_99_percentile,
            }
        )
        scipy.io.savemat(fileName, xOptDict)

    def get_current_e_step_result(self) -> dict:
        return {
            "np_ex_rxyz": self.ex_rxyz,
            "np_ex_txyz": self.ex_txyz,
            "np_focLth": self.focLth,
            "np_dpix": self.dpix,
            "np_u0": self.u0,
            "np_v0": self.v0,
            "np_rela_rxyz": self.rela_rxyz,
            "np_rela_txyz": self.rela_txyz,
            "np_rowScaleXZ": self.rowScaleXZ,
            "np_scales": self.scales,
            "np_rotVecXYZs": self.rotVecXYZs,
            "np_transVecXYZs": self.transVecXYZs,
            "np_featureVec": self.featureVec,
        }

    def load_e_step_result_from_dict(self, xOptDict: dict) -> None:
        self.ex_rxyz = xOptDict["np_ex_rxyz"]
        self.ex_txyz = xOptDict["np_ex_txyz"]
        self.focLth = xOptDict["np_focLth"]
        self.dpix = xOptDict["np_dpix"]
        self.u0 = xOptDict["np_u0"]
        self.v0 = xOptDict["np_v0"]
        self.rela_rxyz = xOptDict["np_rela_rxyz"]
        self.rela_txyz = xOptDict["np_rela_txyz"]
        self.rowScaleXZ = np.ones((2,))
        self.scales = xOptDict["np_scales"]
        self.rotVecXYZs = xOptDict["np_rotVecXYZs"]
        self.transVecXYZs = xOptDict["np_transVecXYZs"]
        self.featureVec = xOptDict["np_featureVec"]

    def load_expectation_step_result(self, filename, stage):
        xOptDict = scipy.io.loadmat(filename, squeeze_me=True)
        self.ex_rxyz = xOptDict["np_ex_rxyz"]
        self.ex_txyz = xOptDict["np_ex_txyz"]
        self.focLth = xOptDict["np_focLth"]
        self.dpix = xOptDict["np_dpix"]
        self.u0 = xOptDict["np_u0"]
        self.v0 = xOptDict["np_v0"]
        self.rela_rxyz = xOptDict["np_rela_rxyz"]
        self.rela_txyz = xOptDict["np_rela_txyz"]
        if stage == 0:
            self.rowScaleXZ = np.ones((2,))
        elif stage == 1:
            self.rowScaleXZ = xOptDict["np_rowScaleXZ"]
        elif stage == 2:
            self.scales = xOptDict["np_scales"]
            self.rotVecXYZs = xOptDict["np_rotVecXYZs"]
            self.transVecXYZs = xOptDict["np_transVecXYZs"]
            self.featureVec = xOptDict["np_featureVec"]

    ###########################################
    ######### Visualization ###################
    ###########################################

    def showEdgeMaskPredictionWithGroundTruth(
        self, photoType, canvasShape=None, dilate=True
    ):
        # red: prediction, white: ground truth
        ph = photoType.value
        if not bool(canvasShape):
            canvasShape = self.edgeMask[ph].shape
        canvas = np.zeros((*canvasShape, 3), dtype=np.float32)
        h, w = self.edgeMask[ph].shape
        canvas[:h, :w, :] = self.edgeMask[ph][:, :, None]  # white: ground truth

        edgePred = np.zeros(canvasShape, dtype=np.float32)
        pix_pred = self.P_pred[ph].astype(np.int32)
        edgePred[pix_pred[:, 1], pix_pred[:, 0]] = 1.0  # red: edge prediction
        if dilate == True:
            edgePred = skimage.morphology.binary_dilation(
                edgePred, skimage.morphology.disk(2)
            )  # dilation edge prediction for visualization
        canvas[:, :, 0] = np.max(np.stack([edgePred, canvas[:, :, 0]]), axis=0)

        plt.figure(figsize=(10, 10))
        plt.imshow(canvas)
        return canvas

    ##################################
    ######### Maximization Step ######
    ##################################

    def anistropicRowScale2ScalesAndTransVecs(self):
        self.scales = np.prod(self.rowScaleXZ) ** (1 / 3) * np.ones_like(
            self.scales, np.float32
        )
        self.transVecXYZs[:, [0, 2]] = self.X_Mu_centroids[:, [0, 2]] * (
            self.rowScaleXZ - 1.0
        )
        self.rowScaleXZ = np.array([1.0, 1.0], np.float32)

    def computePixelResidualError(
        self,
        photoType,
        featureVec,
        scales,
        rotVecXYZs,
        transVecXYZs,
        extrViewMat,
        intrProjMat,
        rela_R,
        rela_txyz,
        rowScaleXZ=np.ones((2,), np.float32),
        stage=1,
        step=-1,
        return_grad=False,
    ):
        # self.X_?_pred: List of array of points in Mu teeth shape, [ndarray1, ndarray2, ...]
        # self.corre_pred_idx: corre indices after vertically stacking the transformed self.X_?_pred

        ph = photoType.value
        tIdx = self.visIdx[ph]
        _corre_pred_idx = self.corre_pred_idx[ph]
        X_deformed_pred = self.X_deformed_pred[ph]
        _X_trans_pred = self.X_deformed_pred[ph]
        X_deformed_pred_normals = self.X_deformed_pred_normals[ph]
        _X_trans_pred_normals = self.X_deformed_pred_normals[ph]

        if stage >= 3:  # consider deformation of tooth shape
            X_deformed_pred = [
                x_mu_pred + np.reshape(sqrtEigVal * fVec @ sigmaTseg, x_mu_pred.shape)
                for x_mu_pred, sqrtEigVal, fVec, sigmaTseg in zip(
                    self.X_Mu_pred[ph],
                    self.SqrtEigVals[tIdx],
                    featureVec,
                    self.SigmaT_segs[ph],
                )
            ]
            X_deformed_pred_normals = self.X_deformed_pred_normals[ph]

        if stage >= 2:  # consider scales and poses of each tooth
            rotMats = self.computeRotMats(rotVecXYZs)
            _X_trans_pred = [
                s * np.matmul(x - tc, R) + t + tc
                for x, s, R, t, tc in zip(
                    X_deformed_pred,
                    scales,
                    rotMats,
                    transVecXYZs,
                    self.X_Mu_centroids[tIdx],
                )
            ]
            _X_trans_pred_normals = [
                np.matmul(xn, R) for xn, R in zip(X_deformed_pred_normals, rotMats)
            ]

        X_trans_pred = deepcopy(_X_trans_pred)
        X_trans_pred_normals = deepcopy(_X_trans_pred_normals)
        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            ul_sp = self.ul_sp[ph]
            X_trans_pred = X_trans_pred[:ul_sp] + [
                x @ rela_R + rela_txyz for x in X_trans_pred[ul_sp:]
            ]
            X_trans_pred_normals = X_trans_pred_normals[:ul_sp] + [
                xn @ rela_R for xn in X_trans_pred_normals[ul_sp:]
            ]

        _X_corre_pred = np.vstack(X_trans_pred)[_corre_pred_idx]
        X_corre_pred = _X_corre_pred.copy()
        if stage == 1:
            X_corre_pred = (
                np.hstack([rowScaleXZ[0], 1.0, rowScaleXZ[1]]) * _X_corre_pred
            )
        X_corre_pred_normals = np.vstack(X_trans_pred_normals)[_corre_pred_idx]

        X_cam_corre_pred = self.updatePointPosInCameraCoord(X_corre_pred, extrViewMat)
        X_cam_corre_pred_normals = self.updatePointNormalsInCameraCoord(
            X_corre_pred_normals, extrViewMat[:3, :]
        )  # extrViewMat.shape = (4,3)

        P_corre_pred = self.updatePointPosInImageCoord(X_cam_corre_pred, intrProjMat)
        P_corre_pred_normals = self.updatePointNormalsInImageCoord(
            X_cam_corre_pred_normals
        )

        errorVecUV = self.P_true_99_percentile[ph] - P_corre_pred  # ci - \hat{ci}
        _M = len(self.P_true_99_percentile[ph])
        resPointError = self.weight_point2point * np.sum(
            np.linalg.norm(errorVecUV, axis=1) ** 2
        )
        resPlaneError = self.weight_point2plane * np.sum(
            np.sum(errorVecUV * P_corre_pred_normals, axis=1) ** 2
        )
        # print("resPointError:{:.4f}, resPlaneError:{:.4f}".format(resPointError/_M, resPlaneError/_M))
        loss = (resPointError + resPlaneError) / _M
        if not return_grad:
            return loss, None

        # gradient with respect to hat_ci and hat_ni
        ci_hatci = errorVecUV  # shape=(_M, 2)
        hatni = P_corre_pred_normals  # shape=(_M, 2)
        ci_hatci_dot_hatni = np.sum(ci_hatci * hatni, axis=1)
        par_loss_par_hatci = (
            -2.0
            / _M
            * np.matmul(
                ci_hatci[:, None, :],
                (
                    self.weight_point2point * np.identity(2, np.float32)
                    + self.weight_point2plane
                    * np.matmul(hatni[:, :, None], hatni[:, None, :])
                ),
            )
        )  # (_M, 1, 2)
        par_loss_par_hatni = (
            2
            * self.weight_point2plane
            / _M
            * ci_hatci_dot_hatni[:, None, None]
            * ci_hatci.reshape(_M, 1, 2)
        )  # (_M, 1, 2)

        g = X_cam_corre_pred  # 3d-point after global transformation (_M,3)
        gn = X_cam_corre_pred_normals
        gz = g[:, [2]]
        gxgy_gz = g[:, :2] / gz  # (_M,2)
        par_hatci_par_fx = 1.0 / self.dpix[ph] * gxgy_gz[..., None]  # (_M,2,1)
        par_hatci_par_u0 = np.array([[1.0], [0.0]], np.float32)
        par_hatci_par_v0 = np.array([[0.0], [1.0]], np.float32)

        # gradient with respect to internal camera params
        grad_fx = np.sum(np.matmul(par_loss_par_hatci, par_hatci_par_fx))
        grad_dpix = 0
        grad_u0 = np.sum(np.matmul(par_loss_par_hatci, par_hatci_par_u0))
        grad_v0 = np.sum(np.matmul(par_loss_par_hatci, par_hatci_par_v0))

        fx = intrProjMat[0, 0]
        par_hatci_par_g = (
            fx
            / gz[:, :, None]
            * np.concatenate(
                [np.tile(np.identity(2, np.float32), (_M, 1, 1)), -gxgy_gz[..., None]],
                axis=-1,
            )
        )  # (_M,2,3)
        par_loss_par_g = np.matmul(par_loss_par_hatci, par_hatci_par_g)  # (_M,1,3)
        # par_g_par_ext = np.identity(3,np.float32)
        par_g_par_exr = -self.skewMatrices(g)

        # gradient with respect to external camera params: ex_rxyz, ex_txyz
        grad_ext = np.sum(par_loss_par_g, axis=0)  # (1,3)
        par_hatni_par_gn = self.jacobs_hatni_wrt_gn(gn)  # (_M,2,3)
        par_loss_par_gn = np.matmul(par_loss_par_hatni, par_hatni_par_gn)  # (_M,1,3)
        par_gn_par_exr = -self.skewMatrices(gn)
        grad_exr = np.sum(np.matmul(par_loss_par_g, par_g_par_exr), axis=0) + np.sum(
            np.matmul(par_loss_par_gn, par_gn_par_exr), axis=0
        )  # (1,3)

        R_global = extrViewMat[:3, :].T
        # gradient  w.r.t relative pose of tooth rows: rela_rxyz, rela_txyz
        rowScaleMat = np.diag([rowScaleXZ[0], 1.0, rowScaleXZ[1]])
        grad_relar = np.zeros((1, 3), np.float32)
        grad_relat = np.zeros((1, 3), np.float32)
        if photoType in [PHOTO.LEFT, PHOTO.RIGHT, PHOTO.FRONTAL]:
            ul_sp = self.ul_sp[ph]
            ks = np.sum([len(x) for x in _X_trans_pred[:ul_sp]]).astype(np.int32)
            idx_l = _corre_pred_idx >= ks

            pl = _X_corre_pred[idx_l]
            pln = X_corre_pred_normals[idx_l]
            par_loss_par_gl = par_loss_par_g[idx_l]
            par_loss_par_gln = par_loss_par_gn[idx_l]
            par_gl_par_relar = np.matmul(R_global @ rowScaleMat, -self.skewMatrices(pl))
            par_gln_par_relar = np.matmul(R_global, -self.skewMatrices(pln))
            par_gl_par_relat = R_global
            grad_relar = np.sum(
                np.matmul(par_loss_par_gl, par_gl_par_relar), axis=0
            ) + np.sum(
                np.matmul(par_loss_par_gln, par_gln_par_relar), axis=0
            )  # (1,3)
            grad_relat = np.sum(
                np.matmul(par_loss_par_gl, par_gl_par_relat), axis=0
            )  # (1,3)

        _Regularizer = 1e-3  # normalize gradient  w.r.t rotation vectors
        grad = np.hstack(
            [
                _Regularizer * np.squeeze(grad_exr),
                np.squeeze(grad_ext),
                grad_fx,
                grad_dpix,
                grad_u0,
                grad_v0,
                _Regularizer * np.squeeze(grad_relar),
                np.squeeze(grad_relat),
            ]
        )

        if stage == 1:  # gradient  w.r.t anistropic scale of tooth row in stage 1
            p = _X_corre_pred
            par_g_par_rowScaleXZ = np.matmul(
                R_global, self.diagMatrices(p)[..., [0, 2]]
            )  # (_M,3,2)
            grad_rowScaleXZ = np.sum(
                np.matmul(par_loss_par_g, par_g_par_rowScaleXZ), axis=0
            )  # (1,2)
            grad = np.hstack([grad, np.squeeze(grad_rowScaleXZ)])

        elif stage == 2:  # gradient  w.r.t scales and poses of each tooth
            numT = self.numTooth
            tIdx = self.visIdx[ph]
            qt_list = [
                x - tc for x, tc in zip(_X_trans_pred, self.X_Mu_centroids[tIdx])
            ]
            qtn_list = _X_trans_pred_normals
            qt = np.vstack(qt_list)[_corre_pred_idx]
            qtn = np.vstack(qtn_list)[_corre_pred_idx]
            _grad_txyzs = np.zeros((numT, 3))
            _grad_rxyzs = np.zeros((numT, 3))
            grad_scales = np.zeros((numT,))
            # par_pu_par_qu = np.eye(3); par_pl_par_ql = rela_R.T
            # par_pnu_par_qnu = np.eye(3); par_pln_par_qln = rela_R.T
            assert len(tIdx) == len(qt_list), "Num of visible teeth should be equal"
            ks = 0
            kt = 0
            for j, tId in enumerate(tIdx):
                ks = copy(kt)
                kt += len(qt_list[j])
                if ks == kt:
                    continue
                mask_j = np.logical_and(_corre_pred_idx >= ks, _corre_pred_idx < kt)
                par_loss_par_pj = np.matmul(par_loss_par_g[mask_j], R_global)  # (?,1,3)
                par_loss_par_pnj = np.matmul(
                    par_loss_par_gn[mask_j], R_global
                )  # (?,1,3)

                if step == 1 or step == 4:
                    # par_qj_par_txyzj = np.identity(3)
                    _grad_txyzs[tId] = par_loss_par_pj.sum(axis=0)
                if step == 2 or step == 4:
                    par_qj_par_rxyzj = -self.skewMatrices(qt[mask_j])  # (?,3,3)
                    par_qnj_par_rxyzj = -self.skewMatrices(qtn[mask_j])  # (?,3,3)
                    _grad_rxyzs[tId] = np.matmul(par_loss_par_pj, par_qj_par_rxyzj).sum(
                        axis=0
                    ) + np.matmul(par_loss_par_pnj, par_qnj_par_rxyzj).sum(axis=0)
                if step == 3 or step == 4:
                    par_qj_par_scalej = qt[mask_j].reshape(-1, 3, 1)
                    grad_scales[tId] = np.matmul(
                        par_loss_par_pj, par_qj_par_scalej
                    ).sum()

            if step == 1:
                grad = np.hstack([grad, _grad_txyzs.flatten()])
            elif step == 2:
                grad = np.hstack([grad, _grad_rxyzs.flatten()])
            elif step == 3:
                grad = np.hstack([grad, grad_scales])
            elif step == 4:
                grad = np.hstack(
                    [grad, _grad_txyzs.flatten(), _grad_rxyzs.flatten(), grad_scales]
                )

        elif stage == 3:  # grad  w.r.t tooth shape vector
            numT = self.numTooth
            tIdx = self.visIdx[ph]
            qs_list = X_deformed_pred
            rotMats = self.computeRotMats(rotVecXYZs)
            # par_g_par_p = R_global # par_p_par_qt = np.eye(3) # par_qt_par_qs = scales * rotMats
            par_g_par_qs = scales[:, None, None] * np.matmul(
                R_global, rotMats
            )  # shape=(len(tIdx),3,3) # par_g_par_qs = par_g_par_p @ par_p_par_qt @ par_qt_par_qs
            _grad_fVecs = np.zeros((self.numTooth, self.numPC))
            assert len(tIdx) == len(qs_list), "Num of visible teeth should be equal"
            ks = 0
            kt = 0
            for j, tId in enumerate(tIdx):
                ks = copy(kt)
                kt += len(qs_list[j])
                if ks == kt:
                    continue
                mask_j = np.logical_and(_corre_pred_idx >= ks, _corre_pred_idx < kt)
                par_loss_par_qsj = np.matmul(
                    par_loss_par_g[mask_j], par_g_par_qs[j]
                )  # (len(idx_j),1,3)
                idx_j = _corre_pred_idx[mask_j] - ks
                sqrtEigVals = np.squeeze(self.SqrtEigVals[tId])  # (numPC,)
                corre_sigmaT_seg = self.SigmaT_segs[ph][j].reshape(self.numPC, -1, 3)[
                    :, idx_j, :, None
                ]  # (numPC, 3*m(j)) -> (numPC, len(idx_j), 3, 1) # m(j) num of pred visible points in tooth-mesh-j
                par_loss_par_fVec = sqrtEigVals * np.squeeze(
                    np.matmul(par_loss_par_qsj[None, ...], corre_sigmaT_seg)
                ).sum(axis=-1)
                _grad_fVecs[tId] = par_loss_par_fVec
            grad = np.hstack([grad, _grad_fVecs.flatten()])
        return loss, grad

    @staticmethod
    def skewMatrices(a):
        # a: 2d-array, shape=(?,3)
        n, _ = a.shape
        vec_0 = np.zeros((n, 1), np.float32)
        vec_a1, vec_a2, vec_a3 = np.split(a, 3, axis=-1)
        return np.stack(
            [vec_0, -vec_a3, vec_a2, vec_a3, vec_0, -vec_a1, -vec_a2, vec_a1, vec_0],
            axis=-1,
        ).reshape((n, 3, 3))

    @staticmethod
    def diagMatrices(a):
        # a: 2d-array, shape=(?,3)
        n, _ = a.shape
        vec_0 = np.zeros((n, 1), np.float32)
        vec_a1, vec_a2, vec_a3 = np.split(a, 3, axis=-1)
        return np.stack(
            [vec_a1, vec_0, vec_0, vec_0, vec_a2, vec_0, vec_0, vec_0, vec_a3], axis=-1
        ).reshape((n, 3, 3))

    @staticmethod
    def jacobs_hatni_wrt_gn(vec_gn):
        # vec_gn.shape = (m, 3), a list of point normals
        m = len(vec_gn)
        vec_gnx = vec_gn[:, 0]
        vec_gny = vec_gn[:, 1]
        vec_0 = np.zeros_like(vec_gnx, np.float32)
        vec_gnx_gny = vec_gnx * vec_gny
        vec_norm_gnxy = np.linalg.norm(vec_gn[:, :2], axis=1, keepdims=True)
        _jacob = np.stack(
            [vec_gny**2, -vec_gnx_gny, vec_0, vec_gnx**2, -vec_gnx_gny, vec_0],
            axis=-1,
        ).reshape(m, 2, 3)
        return 1.0 / (vec_norm_gnxy[:, :, None] ** 3) * _jacob

    def parseGlobalParamsOf5Views(self, params, pIdx):
        ex_rxyz = self.ex_rxyz_lr * params[
            pIdx["ex_rxyz"] : pIdx["ex_rxyz"] + 15
        ].reshape(5, 3)
        ex_txyz = self.ex_txyz_lr * params[
            pIdx["ex_txyz"] : pIdx["ex_txyz"] + 15
        ].reshape(5, 3)
        focLth = self.focLth_lr * params[pIdx["focLth"] : pIdx["focLth"] + 5]
        dpix = self.dpix_lr * params[pIdx["dpix"] : pIdx["dpix"] + 5]
        u0 = self.uv_lr * params[pIdx["u0"] : pIdx["u0"] + 5]
        v0 = self.uv_lr * params[pIdx["v0"] : pIdx["v0"] + 5]
        rela_rxyz = (
            self.rela_rxyz_lr * params[pIdx["rela_rxyz"] : pIdx["rela_rxyz"] + 3]
        )
        rela_txyz = (
            self.rela_txyz_lr * params[pIdx["rela_txyz"] : pIdx["rela_txyz"] + 3]
        )
        return ex_rxyz, ex_txyz, focLth, dpix, u0, v0, rela_rxyz, rela_txyz

    def parseTeethPoseParams(self, params, pIdx, step):
        transVecXYZs = self.transVecXYZs
        rotVecXYZs = self.rotVecXYZs
        scales = self.scales
        numT = self.numTooth
        if step == 1:
            transVecXYZs = params[pIdx["tXYZs"] : pIdx["tXYZs"] + numT * 3].reshape(
                numT, 3
            )
        elif step == 2:
            rotVecXYZs = params[pIdx["rXYZs"] : pIdx["rXYZs"] + numT * 3].reshape(
                numT, 3
            )
        elif step == 3:
            scales = params[pIdx["scales"] : pIdx["scales"] + numT]
        elif step == 4:
            transVecXYZs = params[pIdx["tXYZs"] : pIdx["tXYZs"] + numT * 3].reshape(
                numT, 3
            )
            rotVecXYZs = params[pIdx["rXYZs"] : pIdx["rXYZs"] + numT * 3].reshape(
                numT, 3
            )
            scales = params[pIdx["scales"] : pIdx["scales"] + numT]
        return transVecXYZs, rotVecXYZs, scales

    def getCurrentGlobalParamsOf5Views_as_x0(self, stage, step):
        # stage 0
        pIdx = {
            "ex_rxyz": 0,
            "ex_txyz": 15,
            "focLth": 30,
            "dpix": 35,
            "u0": 40,
            "v0": 45,
            "rela_rxyz": 50,
            "rela_txyz": 53,
        }
        x0 = np.hstack(
            [
                self.ex_rxyz.flatten() / self.ex_rxyz_lr,
                self.ex_txyz.flatten() / self.ex_txyz_lr,
                self.focLth / self.focLth_lr,
                self.dpix / self.dpix_lr,
                self.u0 / self.uv_lr,
                self.v0 / self.uv_lr,
                self.rela_rxyz / self.rela_rxyz_lr,
                self.rela_txyz / self.rela_txyz_lr,
            ]
        )
        # stage 1
        if stage == 1:  # optimize camera params and anistropic scale of tooth row
            pIdx["rowScaleXZ"] = len(x0)
            x0 = np.hstack([x0, self.rowScaleXZ])
        # stage 2
        elif stage == 2:  # optimize scales and poses of teeth
            if step == 1:
                pIdx["tXYZs"] = len(x0)
                x0 = np.hstack([x0, self.transVecXYZs.flatten()])
            elif step == 2:
                pIdx["rXYZs"] = len(x0)
                x0 = np.hstack([x0, self.rotVecXYZs.flatten()])
            elif step == 3:
                pIdx["scales"] = len(x0)
                x0 = np.hstack([x0, self.scales])
            elif step == 4:
                pIdx.update(
                    {
                        "tXYZs": len(x0),
                        "rXYZs": len(x0) + self.numTooth * 3,
                        "scales": len(x0) + self.numTooth * 6,
                    }
                )
                x0 = np.hstack(
                    [
                        x0,
                        self.transVecXYZs.flatten(),
                        self.rotVecXYZs.flatten(),
                        self.scales,
                    ]
                )
        elif stage == 3:  # optimize tooth shape vector
            pIdx["featureVec"] = len(x0)
            x0 = np.hstack([x0, self.featureVec.flatten()])
        return x0, pIdx

    def computeTeethPoseResidualError(
        self, scales, rotVecXYZs, transVecXYZs, tIdx, return_grad=False
    ):
        centeredPoseParams = np.hstack(
            [
                (transVecXYZs - self.meanTransVecXYZs[tIdx]),
                (rotVecXYZs - self.meanRotVecXYZs[tIdx]),
            ]
        )  # shape=(len(tIdx),6)
        A = self.invCovMats[tIdx, :, :]  # shape=(len(tIdx),6,6); A = A.T
        x = centeredPoseParams[:, :, None]  # shape=(len(tIdx),6,1)
        x_T = np.transpose(x, (0, 2, 1))  # shape=(len(tIdx),1,6)
        x_T_times_A = np.matmul(x_T, A)  # shape=(len(tIdx),1,6)
        errorTeethPose = np.sum(np.matmul(x_T_times_A, x))

        centeredScales = scales - self.meanScales[tIdx]
        B = self.invCovMatOfScale[
            tIdx, tIdx[:, None]
        ]  # shape=(len(tIdx),len(tIdx)); B = B.T
        y = centeredScales  # shape=(len(tIdx),)
        y_T_times_B = y @ B
        errorScales = y_T_times_B @ y
        if not return_grad:
            return self.weightTeethPose * (errorTeethPose + errorScales), None
        # gradient of teethPoseError w.r.t tXYZs,rXYZs,scales
        numT = self.numTooth
        _grad_txyzs = np.zeros((numT, 3), np.float32)
        _grad_rxyzs = np.zeros((numT, 3), np.float32)
        grad_scales = np.zeros((numT,), np.float32)
        _grad_txyzs[tIdx] = 2.0 * np.squeeze(x_T_times_A)[:, 0:3]
        _grad_rxyzs[tIdx] = 2.0 * np.squeeze(x_T_times_A)[:, 3:6]
        grad_scales[tIdx] = 2.0 * y_T_times_B
        grad = self.weightTeethPose * np.hstack(
            [_grad_txyzs.flatten(), _grad_rxyzs.flatten(), grad_scales]
        )
        return self.weightTeethPose * (errorTeethPose + errorScales), grad

    def computeFeatureVecResidualError(self, featureVec, tIdx, return_grad=False):
        featureVecError = self.weightFeatureVec * np.sum(featureVec[tIdx] ** 2)
        if not return_grad:
            return featureVecError, None
        _featureVecGrad = np.zeros(featureVec.shape)  # (self.numTooth,1,self.numPC)
        _featureVecGrad[tIdx] = 2.0 * self.weightFeatureVec * featureVec[tIdx]
        return featureVecError, _featureVecGrad.flatten()

    def MStepLoss(self, params, pIdx, stage, step, verbose, return_grad=False):
        (
            ex_rxyz,
            ex_txyz,
            focLth,
            dpix,
            u0,
            v0,
            rela_rxyz,
            rela_txyz,
        ) = self.parseGlobalParamsOf5Views(params, pIdx)
        rowScaleXZ = np.ones((2,))
        transVecXYZs, rotVecXYZs, scales = self.parseTeethPoseParams(params, pIdx, step)
        ex_R = self.computeRotMats(ex_rxyz)
        rela_R = self.updateRelaRotMat(rela_rxyz)
        extrViewMats = np.concatenate([ex_R, ex_txyz[:, None, :]], axis=1)
        featureVec = self.featureVec

        aniScaleError = 0.0
        if stage == 1:
            rowScaleXZ = params[pIdx["rowScaleXZ"] : pIdx["rowScaleXZ"] + 2]
            equiva_scale = np.prod(rowScaleXZ) ** (1 / 3)
            aniScaleError = (
                self.weightAniScale * (equiva_scale - 1.0) ** 2 / self.scaleStd**2
            )
        elif stage == 3:
            featureVec = params[
                pIdx["featureVec"] : pIdx["featureVec"] + self.numTooth * self.numPC
            ].reshape(self.featureVec.shape)

        errors = np.zeros((5,))
        paramNum = len(params)
        M_grad = np.zeros((paramNum,))
        if return_grad == True and stage == 1:
            gradRowScaleXZ = (
                2.0
                * self.weightAniScale
                / (3.0 * self.scaleStd**2)
                * np.prod(rowScaleXZ) ** (-1 / 3)
                * np.array([rowScaleXZ[1], rowScaleXZ[0]])
            )
            M_grad[pIdx["rowScaleXZ"] : pIdx["rowScaleXZ"] + 2] += (
                np.sum(self.weightViews) * gradRowScaleXZ
            )

        for phType in self.photoTypes:
            ph = phType.value
            tIdx = self.visIdx[ph]
            intrProjMat = self.updateIntrinsicProjectionMatrix(
                focLth[ph], dpix[ph], u0[ph], v0[ph]
            )
            pixelError, pixelGrad = self.computePixelResidualError(
                phType,
                featureVec[tIdx],
                scales[tIdx],
                rotVecXYZs[tIdx],
                transVecXYZs[tIdx],
                extrViewMats[ph],
                intrProjMat,
                rela_R,
                rela_txyz,
                rowScaleXZ,
                stage,
                step,
                return_grad,
            )
            teethPoseError = 0.0
            teethPoseGrad = np.zeros((7 * self.numTooth,), np.float32)
            featureVecError = 0.0
            featureVecGrad = np.zeros((self.numPC * self.numTooth,), np.float32)
            if stage == 2:
                teethPoseError, teethPoseGrad = self.computeTeethPoseResidualError(
                    scales[tIdx],
                    rotVecXYZs[tIdx],
                    transVecXYZs[tIdx],
                    tIdx,
                    return_grad,
                )
            elif stage == 3:
                featureVecError, featureVecGrad = self.computeFeatureVecResidualError(
                    featureVec, tIdx, return_grad
                )
            if verbose == True:
                print(
                    "{}, pixelError:{:.2f}, teethPoseError:{:.2f}, featureVecError: {:.2f}".format(
                        str(phType), pixelError, teethPoseError, featureVecError
                    )
                )
            errors[ph] = self.weightViews[ph] * (
                pixelError + teethPoseError + aniScaleError + featureVecError
            )
            if return_grad == True:
                M_grad = self.__updateMStepGradVector(
                    M_grad,
                    pIdx,
                    self.weightViews[ph] * pixelGrad,
                    self.weightViews[ph] * teethPoseGrad,
                    self.weightViews[ph] * featureVecGrad,
                    ph,
                    stage,
                    step,
                )
                # print("gradient of {}".format(str(phType)), self.weightViews[ph]*pixelGrad)

        M_loss = np.sum(errors)
        if verbose == True:
            print(
                "maximization step errors: [{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}]".format(
                    *errors
                )
            )
        if not return_grad:
            return M_loss
        return M_loss, M_grad

    def __updateMStepGradVector(
        self,
        aggGrad,
        pIdx,
        pixelGrad,
        teethPoseGrad,
        featureVecGrad,
        ph,
        stage,
        step=-1,
    ):
        grad_exr = pixelGrad[0:3]
        grad_ext = pixelGrad[3:6]
        grad_fx, grad_dpix, grad_u0, grad_v0 = pixelGrad[6:10]
        grad_relar = pixelGrad[10:13]
        grad_relat = pixelGrad[13:16]

        ex_rxyz_ks = pIdx["ex_rxyz"] + 3 * ph
        ex_txyz_ks = pIdx["ex_txyz"] + 3 * ph
        ex_focLth_ks = pIdx["focLth"] + ph
        ex_dpix_ks = pIdx["dpix"] + ph
        ex_u0_ks = pIdx["u0"] + ph
        ex_v0_ks = pIdx["v0"] + ph
        rela_rxyz_ks = pIdx["rela_rxyz"]
        rela_txyz_ks = pIdx["rela_txyz"]

        aggGrad[ex_rxyz_ks : ex_rxyz_ks + 3] += grad_exr
        aggGrad[ex_txyz_ks : ex_txyz_ks + 3] += grad_ext
        aggGrad[ex_focLth_ks] += grad_fx
        aggGrad[ex_dpix_ks] += grad_dpix
        aggGrad[ex_u0_ks] += grad_u0
        aggGrad[ex_v0_ks] += grad_v0
        aggGrad[rela_rxyz_ks : rela_rxyz_ks + 3] += grad_relar
        aggGrad[rela_txyz_ks : rela_txyz_ks + 3] += grad_relat

        supp_ks = 16
        if stage == 1:
            rowScale_ks = pIdx["rowScaleXZ"]
            grad_rowScaleXZ = pixelGrad[supp_ks : supp_ks + 2]
            aggGrad[rowScale_ks : rowScale_ks + 2] += grad_rowScaleXZ
        elif stage == 2:
            numT = self.numTooth
            if step == 1:
                txyzs_ks = pIdx["tXYZs"]
                aggGrad[txyzs_ks : txyzs_ks + 3 * numT] += (
                    pixelGrad[supp_ks : supp_ks + 3 * numT]
                    + teethPoseGrad[0 : 3 * numT]
                )
            elif step == 2:
                rxyzs_ks = pIdx["rXYZs"]
                aggGrad[rxyzs_ks : rxyzs_ks + 3 * numT] += (
                    pixelGrad[supp_ks : supp_ks + 3 * numT]
                    + teethPoseGrad[3 * numT : 6 * numT]
                )
            elif step == 3:
                scales_ks = pIdx["scales"]
                aggGrad[scales_ks : scales_ks + numT] += (
                    pixelGrad[supp_ks : supp_ks + numT]
                    + teethPoseGrad[6 * numT : 7 * numT]
                )
            elif step == 4:
                txyzs_ks = pIdx["tXYZs"]
                rxyzs_ks = pIdx["rXYZs"]
                scales_ks = pIdx["scales"]
                aggGrad[txyzs_ks : txyzs_ks + 3 * numT] += (
                    pixelGrad[supp_ks : supp_ks + 3 * numT]
                    + teethPoseGrad[0 : 3 * numT]
                )
                aggGrad[rxyzs_ks : rxyzs_ks + 3 * numT] += (
                    pixelGrad[supp_ks + 3 * numT : supp_ks + 6 * numT]
                    + teethPoseGrad[3 * numT : 6 * numT]
                )
                aggGrad[scales_ks : scales_ks + numT] += (
                    pixelGrad[supp_ks + 6 * numT : supp_ks + 7 * numT]
                    + teethPoseGrad[6 * numT : 7 * numT]
                )
        elif stage == 3:
            fVec_ks = pIdx["featureVec"]
            _m = self.numTooth * self.numPC
            aggGrad[fVec_ks : fVec_ks + _m] += (
                pixelGrad[supp_ks : supp_ks + _m] + featureVecGrad
            )
        return aggGrad

    def maximization_step_5Views(self, stage, step, maxiter=100, verbose=True):
        x0, pIdx = self.getCurrentGlobalParamsOf5Views_as_x0(stage, step)
        if stage == 3:
            for phType in self.photoTypes:
                self.SigmaT_segs[phType.value] = self.updateCorreSigmaTSegs(phType)

        # param bounds
        bounds = self.getParamBounds(x0, pIdx, stage, step)
        optRes = scipy.optimize.minimize(
            fun=self.MStepLoss,
            x0=x0,
            jac=True,
            bounds=bounds,
            args=(pIdx, stage, step, False, True),
            method="SLSQP",
            tol=1e-6,
            options={"ftol": 1e-6, "maxiter": maxiter, "disp": False},
        )
        params = optRes.x

        # update params
        (
            self.ex_rxyz,
            self.ex_txyz,
            self.focLth,
            self.dpix,
            self.u0,
            self.v0,
            self.rela_rxyz,
            self.rela_txyz,
        ) = self.parseGlobalParamsOf5Views(params, pIdx)
        if stage == 1:
            self.rowScaleXZ = params[pIdx["rowScaleXZ"] : pIdx["rowScaleXZ"] + 2]
        elif stage == 2:
            self.transVecXYZs, self.rotVecXYZs, self.scales = self.parseTeethPoseParams(
                params, pIdx, step
            )
        elif stage == 3:
            self.featureVec = params[
                pIdx["featureVec"] : pIdx["featureVec"] + self.numTooth * self.numPC
            ].reshape(self.featureVec.shape)
        self.loss_maximization_step = self.MStepLoss(
            params, pIdx, stage, step, verbose, return_grad=False
        )

    def getParamBounds(self, x0, pIdx, stage, step):
        """Get bounds of params"""
        bounds = []
        ex_rxyz_d = 0.3
        ex_txyz_d = 20.0
        ex_rxyz_params = x0[pIdx["ex_rxyz"] : pIdx["ex_rxyz"] + 15]
        ex_txyz_params = x0[pIdx["ex_txyz"] : pIdx["ex_txyz"] + 15]
        rela_rxyz_d = 0.05
        rela_txyz_d = 1.0
        rela_rxyz_params = x0[pIdx["rela_rxyz"] : pIdx["rela_rxyz"] + 3]
        rela_txyz_params = x0[pIdx["rela_txyz"] : pIdx["rela_txyz"] + 3]

        ex_rxyz_bounds = np.stack(
            [ex_rxyz_params - ex_rxyz_d, ex_rxyz_params + ex_rxyz_d]
        )
        ex_rxyz_bounds = list(
            zip(ex_rxyz_bounds[0], ex_rxyz_bounds[1])
        )  # list of tuples
        ex_txyz_bounds = np.stack(
            [ex_txyz_params - ex_txyz_d, ex_txyz_params + ex_txyz_d]
        )
        ex_txyz_bounds = list(
            zip(ex_txyz_bounds[0], ex_txyz_bounds[1])
        )  # list of tuples

        focLth_bounds = [(30.0, 150.0)] * 5
        dpix_bounds = [(None, None)] * 5
        u0_bounds = [(300.0, 500.0)] * 5
        v0_bounds = [(200.0, 400.0)] * 5
        intr_bounds = focLth_bounds + dpix_bounds + u0_bounds + v0_bounds

        rela_rxyz_bounds = [(-rela_rxyz_d, rela_rxyz_d)] * 3
        rela_txyz_bounds = np.stack(
            [rela_txyz_params - rela_txyz_d, rela_txyz_params + rela_txyz_d]
        )
        rela_txyz_bounds = list(
            zip(rela_txyz_bounds[0], rela_txyz_bounds[1])
        )  # list of tuples
        bounds = (
            ex_rxyz_bounds
            + ex_txyz_bounds
            + intr_bounds
            + rela_rxyz_bounds
            + rela_txyz_bounds
        )
        if stage == 1:
            bounds = bounds + [(0.01, 2.0)] * 2  # add bounds for rowScaleXZ
        elif stage == 2:
            numT = self.numTooth
            tXYZs_d = 10.0 * self.transVecStd
            rXYZs_d = 4.0 * self.rotVecStd
            scales_d = 4.0 * self.scaleStd
            if step == 1:
                bounds += [(-tXYZs_d, tXYZs_d)] * (
                    3 * numT
                )  # add bounds for tooth translation vecs
            elif step == 2:
                bounds += [(-rXYZs_d, rXYZs_d)] * (
                    3 * numT
                )  # add bounds for tooth rot vecs
            elif step == 3:
                bounds += [
                    (1.0 - scales_d, 1.0 + scales_d)
                ] * numT  # add bounds for tooth scales
            elif step == 4:
                bounds += (
                    [(-tXYZs_d, tXYZs_d)] * (3 * numT)
                    + [(-rXYZs_d, rXYZs_d)] * (3 * numT)
                    + [(1.0 - scales_d, 1.0 + scales_d)] * numT
                )
        elif stage == 3:
            bounds += [(-5.0, 5.0)] * (
                self.numTooth * self.numPC
            )  # add bounds for featureVec (mean=0,std=1)
        return bounds

    ###########################################
    ######### Maximization stage 3 ############
    ###########################################

    def updateCorreSigmaTSegs(self, photoType):
        ph = photoType.value
        tIdx = self.visIdx[ph]
        SigmaT_segs = []
        for sigmaT, vis_hull_vids in zip(
            self.SigmaT[tIdx], self.vis_hull_vertex_indices[ph]
        ):  # self.SigmaT.shape=(numTooth,numPC,numPoint*3)
            sigmaTseg = sigmaT.reshape(self.numPC, self.numPoint, 3)[
                :, vis_hull_vids, :
            ]
            SigmaT_segs.append(sigmaTseg.reshape(self.numPC, 3 * len(vis_hull_vids)))
        return SigmaT_segs

    ###################################
    ######### Save H5 File ############
    ###################################

    def saveDemo2H5(self, h5File):
        if not os.path.exists(os.path.dirname(h5File)):
            os.makedirs(os.path.dirname(h5File))
        with h5py.File(h5File, "w") as f:  # ovewrite file each time
            grp = f.create_group("EMOPT")
            grp.create_dataset(
                "UPPER_INIT",
                data=np.array(self.X_Mu[: self.numUpperTooth], dtype=np.double),
            )
            grp.create_dataset(
                "LOWER_INIT",
                data=np.array(self.X_Mu[self.numUpperTooth :], dtype=np.double),
            )
            grp.create_dataset(
                "UPPER_PRED",
                data=np.array(self.X_trans[: self.numUpperTooth], dtype=np.double),
            )
            grp.create_dataset(
                "LOWER_PRED",
                data=np.array(self.X_trans[self.numUpperTooth :], dtype=np.double),
            )
            grp.create_dataset("MASK", data=np.array(self.Mask, dtype=np.double))
            grp.create_dataset("RELA_R", data=np.array(self.rela_R, dtype=np.double))
            grp.create_dataset("RELA_T", data=np.array(self.rela_txyz, dtype=np.double))
            grp.create_dataset("EX_RXYZ", data=np.array(self.ex_rxyz, dtype=np.double))
            grp.create_dataset("EX_TXYZ", data=np.array(self.ex_txyz, dtype=np.double))
            grp.create_dataset("FOCLTH", data=np.array(self.focLth, dtype=np.double))
            grp.create_dataset("DPIX", data=np.array(self.dpix, dtype=np.double))
            grp.create_dataset("U0", data=np.array(self.u0, dtype=np.double))
            grp.create_dataset("V0", data=np.array(self.v0, dtype=np.double))
            grp.create_dataset("SCALES", data=np.array(self.scales, dtype=np.double))
            grp.create_dataset(
                "ROT_VEC_XYZS", data=np.array(self.rotVecXYZs, dtype=np.double)
            )
            grp.create_dataset(
                "TRANS_VEC_XYZS", data=np.array(self.transVecXYZs, dtype=np.double)
            )
            grp.create_dataset(
                "FEATURE_VEC", data=np.array(self.featureVec, dtype=np.double)
            )

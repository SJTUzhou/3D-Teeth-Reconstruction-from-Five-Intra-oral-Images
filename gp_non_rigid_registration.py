import numpy as np
import scipy
import scipy.optimize as opt
# from cyipopt import minimize_ipopt

class GP_Non_Rigid_Registration(object):
    def __init__(self, s, sigma, srcX, targetY=None, n=21) -> None:
        # use gaussian kernel g(x',x) = s * exp(-||x'-x||^2 / sigma^2)
        self.s = s 
        self.sigma = sigma
        assert srcX.ndim == 2, "srcX.ndim != 2"
        self.X = srcX # source pointcloud
        self.N = self.X.shape[0]
        self.d = self.X.shape[1]
        self.Y = targetY # target pointcloud
        # self.GP_Mu = np.zeros((self.N*self.d,))

        # 待计算
        self.n = n
        self.lmbda_n = None # the first n eigen values
        self.phi_n = None # the first n eigen vectors (approx. of eigen functions)
        self.alpha = None
        self.X_deformed = None

        
    def compute_EigVals_EigFuncs(self):
        # compute GP K
        distMat = scipy.spatial.distance_matrix(self.X, self.X, p=2, threshold=int(1e8)) # If M * N * K > threshold, algorithm uses a Python loop instead of large temporary arrays.
        GP_K = self.s * np.kron(np.exp( -distMat**2 / self.sigma**2), np.identity(self.d))
        # Nyström Method 计算矩阵GP_K的特征值，特征向量用于近似GP_K的特征值和特征函数 low-rank approx. of GP
        eigVal, u = scipy.linalg.eigh(GP_K) # np.linalg.eig(self.GP_K)
        eigOrder = sorted(range(len(eigVal)), key=lambda x:eigVal[x], reverse=True) # eigVal从大到小的索引排序
        eigVal[eigVal<1e-4] = 0.
        self.lmbda_n = eigVal[eigOrder[:self.n]]
        self.phi_n = u[:,eigOrder[:self.n]]  # * np.sqrt(self.N)
        print("GP low-rank approx. cumulative variance: {:.4f}".format(np.sum(self.lmbda_n)/np.sum(eigVal)))


    def setTargetPcl(self, targetY):
        self.Y = targetY

    @staticmethod
    def chamferDistance(X1, X2): # 倒角距离 量纲[m^2]
        squaredDistMat = scipy.spatial.distance_matrix(X1, X2, p=2, threshold=int(1e8)) ** 2
        return np.min(squaredDistMat, axis=0).mean() + np.min(squaredDistMat, axis=1).mean()


    def loss(self, alpha, eta):
        normalization = eta * np.sum(alpha**2)
        GP = np.sum(alpha * self.lmbda_n * self.phi_n, axis=1) # + self.GP_Mu # use GP to model deformation, ignoring GP_Mu = [0]
        X_deformed = self.X + GP.reshape(self.N, self.d)
        chamfer_dist = self.chamferDistance(X_deformed, self.Y)
        # print("{:.4f}".format(chamfer_dist))
        return chamfer_dist + normalization

    def register(self, eta):
        alpha = np.zeros((self.n,))
        # ret = opt.minimize(self.loss, alpha, args=(eta,), method="Nelder-Mead", options={"fatol":1e-3,"maxiter":100})
        # ret = minimize_ipopt(self.loss, alpha, args=(eta,), options={"maxiter":20,"print_info_string":"yes"})
        ret = opt.minimize(self.loss, alpha, args=(eta,), jac="2-point", method="SLSQP", options={"ftol":1e-6,"maxiter":30,"disp":False})
        self.alpha = ret.x
        GP = np.sum(self.alpha * self.lmbda_n * self.phi_n, axis=1) # + self.GP_Mu  # ignoring GP_Mu = [0]
        self.X_deformed = self.X + np.reshape(GP, self.X.shape)
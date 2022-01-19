import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def phi(Y, mu_k, Sigma_k):
    """calculate the Gaussian density function of the k-th model

    :param Y :dataset
    :param mu_k: the mu of k-th model
    :param Sigma_k: the Sigma of k-th model
    :return : the probability of the sample occured in the model
    :return type : list
    """
    norm = multivariate_normal(mean=mu_k, cov=Sigma_k)  # 计算多元正态随机变量
    return norm.pdf(Y)


def getExpectation(Y, mu, Sigma, pi):
    """E step

    :param Y : data matrix
    :param mu: the mean of each characterristic of each sample ; mu is a 3*4 matrix
    :param Sigma :three-covariance-matrix list
    :param pi: the responsibilities array
    :return : the new responsibilities matrix(gamma)
    :return type : matrix
    """
    # 样本数
    N = Y.shape[0]
    # 模型数
    K = pi.shape[0]

    # 响应度矩阵，行对应样本，列对应响应度
    gamma = np.mat(np.zeros((N, K)))

    # 计算各模型中所有样本出现的概率，行对应样本，列对应模型
    prob = np.zeros((N, K))
    for k in range(K):
        prob[:, k] = phi(Y, mu[k], Sigma[k])
    prob = np.mat(prob)

    # 计算每个模型对每个样本的响应度
    # TODO
    return gamma


def maximize(Y, gamma):
    """M step

    :param Y: data matrix
    :param gamma : the responsibilities matrix
    :return : the parameters : mu, gamma, pi

    """
    # 样本数和特征数
    N, D = Y.shape
    # 模型数
    K = gamma.shape[1]

    # 初始化参数值
    mu = np.zeros((K, D))
    Sigma = []
    pi = np.zeros(K)

    # 更新每个模型的参数
    for k in range(K):
        # 第 k 个模型对所有样本的响应度之和
        Nk = np.sum(gamma[:, k])
        # 更新 mu
        # 对每个特征求均值
        # TODO
        # 更新 Sigma
        # TODO
        # 更新 pi
        # TODO
    Sigma = np.array(Sigma)
    return mu, Sigma, pi


def scale_data(Y):
    """ "scale the data between 0 and 1

    :param Y :dataset
    :return : the scaled dataset

    """
    for i in range(Y.shape[1]):  # the column of dataset
        max_ = Y[:, i].max()
        min_ = Y[:, i].min()
        Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
    return Y


def init_params(shape, K):
    """initialize the parameters : mu, gamma, pi

    :param shape: the row and column of data
    :param K: the number of model
    :return : the initial parameters

    """
    N, D = shape
    mu = np.random.rand(K, D)
    Sigma = np.array([np.eye(D)] * K)
    pi = np.array([1.0 / K] * K)
    return mu, Sigma, pi


def GMM_EM(Y, K, times):
    """GMM_EM

    :param Y :dataset
    :param K ：the number of model (3)
    :param times : the iteration times
    :return : the parameters of three models - mu, gamma , pi

    """
    Y = scale_data(Y)
    mu, Sigma, pi = init_params(Y.shape, K)
    for i in range(times):
        gamma = getExpectation(Y, mu, Sigma, pi)
        mu, Sigma, pi = maximize(Y, gamma)
    return mu, Sigma, pi, gamma


def loadData(filename):
    """从文件中读取数据

    :param filename : the path of file
    :return : the dataset
    :return type : list

    """
    dataSet = []
    with open(filename) as fr:
        for i, line in enumerate(fr.readlines()):
            curLine = line.strip().split(",")
            fltLine = list(map(float, curLine[:-1]))
            dataSet.append(fltLine)
    return dataSet


# 载入数据
Y = loadData("irisdata.txt")
matY = np.matrix(Y, copy=True)

# 计算 GMM 模型参数
mu, Sigma, pi, gamma = GMM_EM(matY, 3, 100)
print("******mu******")
print(mu)
print("*****Sigma******")
print(Sigma)
print("******pi******")
print(pi)
print("******gamma******")
print(gamma)

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:04:30 2024

@author: erika
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

PI2 = math.pi**2

class Newton_method():
    def __init__(self, max_iter=100, tol=1e-10, epsilon=1e-10):
        self.max_iter = max_iter
        self.tol = tol
        self.epsilon = epsilon
    def NewtonIteration(self, Function, InitialValue, t_values):
        matrix = InitialValue.copy()
        N = len(InitialValue)
        Range = len(t_values)
        Matrix = np.zeros((Range,N))
        for i in range(Range):
            for _ in range(self.max_iter):
                # 计算Function的值和雅可比矩阵J
                F = Function(matrix, t_values[i])
                J = self.Jacobian(Function, matrix, t_values[i])
                # 求解线性方程组 J*delta = -F， 若J不可逆，则尝试使用伪逆来代替求解矩阵的逆
                try:
                    delta = np.linalg.solve(J, -F)
                except np.linalg.LinAlgError:
                    delta = np.linalg.pinv(J) @ (-F)
                matrix += delta
                if np.linalg.norm(delta) < self.tol:
                    break
            Matrix[i] = matrix
        return Matrix
    def Jacobian(self, Function, Value, t_value):
        N = len(Value)
        Matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Matrix[i][j] = self.PartialDerivative(Function, Value, t_value, i, j)
        return Matrix   
    def PartialDerivative(self, Function, Value, t_value, i, j):
        N = len(Value)
        Delta = np.zeros(N)
        Delta[j] = self.epsilon
        df_i_j = (Function(Value+Delta, t_value)[i] - Function(Value, t_value)[i]) / self.epsilon
        return df_i_j
      

class DDME_Model():
    def __init__(self):
        # 定义模型参数
        self.M_N = 939.0  # 质子和中子的质量（单位：MeV/c^2）
        self.M_P = 939.0
        self.M_e = 0.511  # 电子质量
        self.M_mu = 105.65839 # mu子质量
        self.rho_B0 = 0.152
        self.HC = 197.327 # from MeV to fm^-1
        self.HC4 = self.HC**4
        # 对称因子
        self.S_delta = None
        self.S_gamma = None
        # beta-平衡
        self.equilibrium = None
        # 化学势
        self.mu_e = None
        self.mu_p = None
        self.mu_n = None
        # 耦合系数
        self.coupling_sigma = None
        self.coupling_omega = None
        self.coupling_rho   = None
        # 密度项
        self.Rho_B = None
        self.Rho_s = None

    def DDME_parameters(self, parameters):
        m_sigma, m_omega, m_rho, \
        Gamma_sigma0, Gamma_omega0, Gamma_rho0, \
        Gamma_sigma_a, Gamma_sigma_b, Gamma_sigma_c, Gamma_sigma_d, \
        Gamma_omega_a, Gamma_omega_b, Gamma_omega_c, Gamma_omega_d, \
        Gamma_rho_a = parameters
        self.M_sigma = m_sigma  # σ介子的质量（单位：MeV/c^2）
        self.M_omega = m_omega  # ω介子的质量（单位：MeV/c^2）
        self.M_rho   = m_rho        
        self.GRS0 = Gamma_sigma0   # Gamma_sigma(rho_B0)
        self.GRW0 = Gamma_omega0   # Gamma_omage(rho_B0)
        self.GRR0 = Gamma_rho0     # Gamma_rho(rho_B0)
        self.GSA = Gamma_sigma_a   # f(x) = a*(1+b*(x+d)**2)/(1+c*(x+d)**2)
        self.GSB = Gamma_sigma_b
        self.GSC = Gamma_sigma_c
        self.GSD = Gamma_sigma_d
        self.GWA = Gamma_omega_a   # f(x) = a*(1+b*(x+d)**2)/(1+c*(x+d)**2)
        self.GWB = Gamma_omega_b
        self.GWC = Gamma_omega_c
        self.GWD = Gamma_omega_d
        self.GRA = Gamma_rho_a     # f(x) = exp(-a*(x-1))
        # 单位变换
        self.M_N /= self.HC
        self.M_P /= self.HC
        self.M_sigma /= self.HC
        self.M_omega /= self.HC
        self.M_rho /= self.HC
        self.M_e /= self.HC
        self.M_mu /= self.HC
    
    # 对称度和beta-平衡
    def SymmetryAndEquilibrium(self, equilibrium = 0, delta = 0):
        self.equilibrium = equilibrium
        self.S_delta = delta    
    
    # 运动方程 
    def EquationsOfMotion(self, IV, RHOB):  
        self.sigma, self.omega, self.rho, self.KF_P, KF = IV # 初值
        # 代入RHOB
        x = RHOB / self.rho_B0
        self.Coupling(x)
        # equation
        if self.equilibrium == 0:
            self.KF_N = KF
            self.Density(RHOB)
            eq4 = self.Rho_B - RHOB       # Rho_B(N)+Rho_B(P) = RHOB
            eq5 = self.Rho_B_N/RHOB - (1-self.S_delta)/2 # 对称度
        elif self.equilibrium == 1:
            self.KF_e = KF
            self.Density(RHOB)
            self.MuThresholdDensity() # 计算电子和mu子
            self.mu_p = self.ChemicalPotential(x, 'p')
            self.mu_n = self.ChemicalPotential(x, 'n')
            # 电荷平衡条件 \rho_Bp = \rho_e + \rho_\mu
            eq4 = self.Rho_B_P - (self.Rho_B_e + self.Rho_B_mu)  
            # beta平衡条件 \mu_e = \mu_n - \mu_p
            eq5 = self.mu_p + self.mu_e - self.mu_n  
        eq1 = self.M_sigma**2 * self.sigma - self.coupling_sigma*self.Rho_s
        eq2 = self.M_omega**2 * self.omega - self.coupling_omega*self.Rho_B
        eq3 = self.M_rho**2 * self.rho - 1/2*self.coupling_rho*self.Rho3
        return np.array([eq1, eq2, eq3, eq4, eq5])
    
    # 判断mu子是否存在，需\mu_e > m_\mu
    def MuThresholdDensity(self):
        self.mu_e = math.sqrt(self.M_e**2 + self.KF_e**2)
        self.Rho_B_e = 2/(6*PI2) * self.KF_e**3
        if self.mu_e < self.M_mu:
            self.KF_mu = 0
            self.Rho_B_mu = 0
        else:
            self.KF_mu = math.sqrt(self.mu_e**2 - self.M_mu**2)
            self.Rho_B_mu = 2/(6*PI2) * self.KF_mu**3
    # 中子和质子的化学势
    def ChemicalPotential(self, x, Baryon):
        U1 = self.coupling_omega * self.omega
        U2 = self.coupling_rho * self.rho  / 2
        if Baryon == 'p':
            M, KF = self.M_P, self.KF_P
            U = U1+U2
        elif Baryon == 'n':
            M, KF = self.M_N, self.KF_N
            U = U1-U2
        MS = M - self.coupling_sigma * self.sigma
        EFS = math.sqrt(KF**2 + MS**2)
        Sigma_R = self.Sigma_R(x) 
        return EFS + U + Sigma_R
    def FermiMomentum(self, RhoB):
        return (3 * PI2 * RhoB ) ** (1/3)

    # 耦合函数及偏导
    def Gamma_function(self, x, gamma0, a, b, c, d): # x = rho_B / rho_B0
        return gamma0 * a * (1 + b*(x+d)**2) / (1 + c*(x+d)**2)    
    def Gamma_partial_function(self, x, gamma0, a, b, c, d):
        return 2 / self.rho_B0 * gamma0 * a * (b-c) *(x+d) / (1 + c*(x+d)**2)**2    

    def Gamma_exp(self, x, gamma0, a):
        return gamma0 * math.exp(-a*(x-1))    
    def Gamma_partial_exp(self, x, gamma0, a):
        return - a / self.rho_B0 * self.Gamma_exp(x, gamma0, a)
    
    # meson-nucleon coupling 
    def Coupling(self, x):
        self.coupling_sigma = self.Gamma_function(x, self.GRS0, self.GSA, self.GSB, self.GSC, self.GSD)
        self.coupling_omega = self.Gamma_function(x, self.GRW0, self.GWA, self.GWB, self.GWC, self.GWD)
        self.coupling_rho   = self.Gamma_exp(x, self.GRR0, self.GRA)     
    # 密度项
    def CalculateDensity(self, M, Rho_B):
        MS = M - self.coupling_sigma * self.sigma
        KF = self.FermiMomentum(Rho_B)
        EFS = math.sqrt(KF**2 + MS**2)
        Rho_s = 2/(4*PI2) * MS * (KF*EFS - MS**2*math.log((KF+EFS)/abs(MS)) )
        return MS, Rho_s, KF
    def Density(self, RHOB):
        self.Rho_B_P = 2/(6*PI2) * self.KF_P**3
        if self.equilibrium == 0: # 避免无beta-平衡情况下，求解方程组中出现0
            self.Rho_B_N = 2/(6*PI2) * self.KF_N**3
        else:
            self.Rho_B_N = RHOB - self.Rho_B_P
        self.MS_P, self.Rho_s_P, self.KF_P = self.CalculateDensity(self.M_P, self.Rho_B_P)
        self.MS_N, self.Rho_s_N, self.KF_N = self.CalculateDensity(self.M_N, self.Rho_B_N)
        # rho_s, rho_s3, rho3
        self.Rho_s = self.Rho_s_P + self.Rho_s_N
        self.Rho_s3 = self.Rho_s_P - self.Rho_s_N
        self.Rho_B = self.Rho_B_P + self.Rho_B_N
        self.Rho3 = self.Rho_B_P - self.Rho_B_N
    
    # 重排项
    def Sigma_R(self, x):
        GP_sigma = self.Gamma_partial_function(x, self.GRS0, self.GSA, self.GSB, self.GSC, self.GSD)
        GP_omega = self.Gamma_partial_function(x, self.GRW0, self.GWA, self.GWB, self.GWC, self.GWD)
        GP_rho   = self.Gamma_partial_exp(x, self.GRR0, self.GRA) 
        return - GP_sigma*self.sigma*self.Rho_s + GP_omega*self.omega*self.Rho_B + \
                 GP_rho*self.rho*self.Rho3/2

    def Ekin(self, MS, KF): # M*, Kf
        EFS = math.sqrt(KF**2 + MS**2)
        return (2 / 16.0 / PI2) * ((2.0 * KF**3 +  MS**2 * KF) * EFS -
                                     MS**4 * math.log((KF + EFS) / abs(MS)))
    def Pkin(self, MS, KF):
        EFS = math.sqrt(KF**2 + MS**2)
        return (2 / 48.0 / PI2) * ((2.0 * KF**3 - 3.0 * MS**2 * KF) * EFS + 
                                     3.0 * MS**4 * math.log((KF + EFS) / abs(MS)))
    # 总能量与总压强
    def EnergyTotal(self):
        US = self.M_sigma**2 * self.sigma**2 / 2
        EW = self.coupling_omega * self.omega * self.Rho_B
        UW = self.M_omega**2 * self.omega**2 / 2
        ER = self.coupling_rho * self.rho * self.Rho3 / 2
        UR = self.M_rho**2 * self.rho**2 /2
        Ek_P = self.Ekin(self.MS_P, self.KF_P)
        Ek_N = self.Ekin(self.MS_N, self.KF_N)
        if self.equilibrium == 0:
            return (Ek_P + Ek_N) + US - UW - UR + EW + ER
        else:
            Ek_e = self.Ekin(self.M_e, self.KF_e)
            Ek_mu = self.Ekin(self.M_mu, self.KF_mu)
            return (Ek_P + Ek_N + Ek_e + Ek_mu) + US - UW - UR + EW + ER
    def PressureTotal(self, x):
        US = self.M_sigma**2 * self.sigma**2 / 2
        UW = self.M_omega**2 * self.omega**2 / 2
        UR = self.M_rho**2 * self.rho**2 /2       
        Pk_P = self.Pkin(self.MS_P, self.KF_P)
        Pk_N = self.Pkin(self.MS_N, self.KF_N)  
        Rearrangement = self.Sigma_R(x) * self.Rho_B
        if self.equilibrium == 0:
            return (Pk_P + Pk_N) - US + UW + UR + Rearrangement
        else:
            Pk_e = self.Pkin(self.M_e, self.KF_e)
            Pk_mu = self.Pkin(self.M_mu, self.KF_mu)
            return (Pk_P + Pk_N + Pk_e + Pk_mu) - US + UW + UR + Rearrangement
    
    # 对称能
    def SymmetryEnergy(self):
        KF = self.FermiMomentum(self.Rho_B/2)
        EFS = math.sqrt(KF**2 + self.MS_N**2)
        Esym = KF**2/6/EFS + self.coupling_rho**2*self.Rho_B/8/self.M_rho**2
        return Esym
    
    # 对称能斜率
    def SymmetryEnergySlope(self, E, rho_B):
        N = len(rho_B)-2
        Esls = []
        for i in range(N):
            I = i+1
            h = (rho_B[I+1]-rho_B[I-1])/2
            dE_dR = (E[I+1]-E[I-1])/(2*h)
            esls = 3*rho_B[I]*dE_dR
            Esls.append(esls)
        return rho_B[1:-1], np.array(Esls)
    # 不可压缩系数
    def Imcompressibility(self, P, rho_B):
        N = len(rho_B)-2
        K = []
        for i in range(N):
            I = i+1
            h = (rho_B[I+1]-rho_B[I-1])/2
            dP_dR = (P[I+1]-P[I-1])/(2*h)
            K.append(9*dP_dR)
        return rho_B[1:-1], np.array(K)
    # 核物质性质
    def GetProperties(self, Value, RHOB):
        self.sigma, self.omega, self.rho, self.KF_P, KF = Value # 初值
        # 代入RHOB
        x = RHOB / self.rho_B0
        self.Coupling(x)
        # equation
        if self.equilibrium == 0:
            self.KF_N = KF
            self.Density(RHOB)
        elif self.equilibrium == 1:
            self.KF_e = KF
            self.Density(RHOB)
            self.MuThresholdDensity() # 计算电子和mu子
        # 性质
        MSN = self.M_N - self.coupling_sigma * self.sigma
        Etot = self.EnergyTotal()
        Ptot = self.PressureTotal(x)
        EA = Etot/RHOB - self.M_N
        Esym = self.SymmetryEnergy()
        return MSN, Etot*self.HC, Ptot*self.HC, EA*self.HC, Esym*self.HC
    # 
    def ChemicalPotentialOfParticles(self, Values, RHOB):
        self.sigma, self.omega, self.rho, self.KF_P, self.KF_e = Values # 初值
        # 代入RHOB
        x = RHOB / self.rho_B0
        self.Coupling(x)
        self.Density(RHOB)
        self.MuThresholdDensity() # 计算电子和mu子
        self.mu_p = self.ChemicalPotential(x, 'p')
        self.mu_n = self.ChemicalPotential(x, 'n')
        return self.mu_p, self.mu_n, self.mu_e
    def RatioOfDensity(self, Values, RHOB):
        self.sigma, self.omega, self.rho, self.KF_P, self.KF_e = Values # 初值
        # 代入RHOB
        x = RHOB / self.rho_B0
        self.Coupling(x)
        self.Density(RHOB)        
        self.MuThresholdDensity() # 计算电子和mu子
        RHO = self.Rho_B_P + self.Rho_B_N + self.Rho_B_e + self.Rho_B_mu
        return self.Rho_B_P/RHO, self.Rho_B_N/RHO, self.Rho_B_e/RHO, self.Rho_B_mu/RHO
    

# 参数
IRHOB = 2001
RHOB = np.linspace(0.001,2.001,IRHOB) 
InitialValues = [0.01, 0.01, 0.01, 0.01, 0.01] # 求解初值
DDMEX_parameters = [547.3327, 783, 763, 10.7067, 13.3388, 7.2380, 1.3970, 1.3350,
                    2.0671, 0.4016, 1.3936, 1.0191, 1.6060, 0.4556, 0.6202]

# 初始化模型
DDMEX = DDME_Model()
DDMEX.DDME_parameters(DDMEX_parameters)
DDMEX.SymmetryAndEquilibrium(equilibrium = 1)

# 牛顿迭代法求解运动方程
Newton = Newton_method()
Equations_solve_DDMEX = Newton.NewtonIteration(DDMEX.EquationsOfMotion, InitialValues, RHOB)
sigma, omega, rho, KF_P, KF_N = Equations_solve_DDMEX.transpose()


# 计算核物质性质
Properties = np.zeros((IRHOB,5))
for i in range(IRHOB):
    result = DDMEX.GetProperties(Equations_solve_DDMEX[i], RHOB[i])
    Properties[i] = result
MSN_DDMEX, Etot_DDMEX, Ptot_DDMEX, EA_DDMEX, Esym_DDMEX = Properties.transpose()
#Esls = DDMEX.SymmetryEnergySlope(Esym, RHOB)
#K = DDMEX.Imcompressibility(Ptot, RHOB)

# 饱和点性质
def GetY(x,y,x0):
    k = (y[1]-y[0])/(x[1]-x[0])
    return y[0] + k*(x0-x[0])
def SaturationPointProperties(RHOB, EA, K, Esym, Esls, MS, point):
    for i in range(len(RHOB)):
        if RHOB[i] <= point and point <= RHOB[i+1]:
            x = RHOB[i:i+2]
            EA0   = GetY(x, EA[i:i+2]    , point)
            K0    = GetY(x, K[i+1:i+3]   , point)
            Esym0 = GetY(x, Esym[i:i+2]  , point)
            Esls0 = GetY(x, Esls[i+1:i+3], point)
            MS0   = GetY(x, MS[i:i+2]    , point)
            break
    return point, EA0, K0, Esym0, Esls0, MS0/DDMEX.M_N
# SSP = SaturationPointProperties(RHOB, EA, K[1], Esym, Esls[1], MSN, 0.152)

###
# DDME2
DDME2_parameters = [550.1238, 783, 763, 10.5396, 13.0189, 7.3672, 1.3881, 1.0943,
                    1.7057, 0.4421, 1.3892, 0.9240, 1.4620, 0.4775, 0.5647]
DDME2 = DDME_Model()
DDME2.SymmetryAndEquilibrium(equilibrium = 1)
DDME2.DDME_parameters(DDME2_parameters)
Equations_solve_DDME2 = Newton.NewtonIteration(DDME2.EquationsOfMotion, InitialValues, RHOB)
Properties = np.zeros((IRHOB,5))
for i in range(IRHOB):
    result = DDME2.GetProperties(Equations_solve_DDME2[i], RHOB[i])
    Properties[i] = result
MSN_DDME2, Etot_DDME2, Ptot_DDME2, EA_DDME2, Esym_DDME2 = Properties.transpose()


# beta-平衡性质
# 化学势
ChemicalPotential = np.zeros((IRHOB, 3))
for i in range(IRHOB):
    result = DDMEX.ChemicalPotentialOfParticles(Equations_solve_DDMEX[i], RHOB[i])
    ChemicalPotential[i] = result
mu_p_DDMEX, mu_n_DDMEX, mu_e_DDMEX = ChemicalPotential.transpose()
#
ChemicalPotential = np.zeros((IRHOB, 3))
for i in range(IRHOB):
    result = DDME2.ChemicalPotentialOfParticles(Equations_solve_DDME2[i], RHOB[i])
    ChemicalPotential[i] = result
mu_p_DDME2, mu_n_DDME2, mu_e_DDME2 = ChemicalPotential.transpose()
# 密度占比
Ratio = np.zeros((IRHOB, 4))
for i in range(IRHOB):
    result = DDMEX.RatioOfDensity(Equations_solve_DDMEX[i], RHOB[i])
    Ratio[i] = result
rp_DDMEX, rn_DDMEX, re_DDMEX, rmu_DDMEX = Ratio.transpose()
#
Ratio = np.zeros((IRHOB, 4))
for i in range(IRHOB):
    result = DDME2.RatioOfDensity(Equations_solve_DDME2[i], RHOB[i])
    Ratio[i] = result
rp_DDME2, rn_DDME2, re_DDME2, rmu_DDME2 = Ratio.transpose()

# plot
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
'''边框设定'''
def fig_ax(size):
    fig,ax = plt.subplots(figsize=size)
    ax.tick_params(top='on', right='on', which='major',width=4.5 ,length=20 ,labelsize=35,pad=10)
    ax.tick_params(top='on', right='on', which='minor',width=2 ,length=12 ,labelsize=35,pad=10)
    ax.spines['left'].set_linewidth('4.5')
    ax.spines['right'].set_linewidth('4.5')
    ax.spines['bottom'].set_linewidth('4.5')
    ax.spines['top'].set_linewidth('4.5') 
    return fig,ax
def ax_rho_e(ax):
    ax.set_yscale('log')
    ax.set_xlim(0,2.0)
    ax.set_ylim(1e1,1e4)
    ax.set_xlabel(r'$\rho_B\ $[fm$^{-3}$]',fontsize=50)
    ax.set_ylabel(r'$\epsilon\ $[MeV/fm$^{-3}$]',fontsize=50)
    xmajorLocator = MultipleLocator(0.5)  
    ax.xaxis.set_major_locator(xmajorLocator)
    xminorLocator = ticker.MultipleLocator(0.1) 
    ax.xaxis.set_minor_locator(xminorLocator)
def ax_rho_p(ax):
    ax.set_yscale('log')
    ax.set_xlim(0,2.0)
    ax.set_ylim(1e-2,1e4)
    ax.set_xlabel(r'$\rho_B\ $[fm$^{-3}$]',fontsize=50)
    ax.set_ylabel(r'P$\ $[MeV/fm$^{-3}$]',fontsize=50)
    xmajorLocator = MultipleLocator(0.5)  
    ax.xaxis.set_major_locator(xmajorLocator)
    xminorLocator = ticker.MultipleLocator(0.1) 
    ax.xaxis.set_minor_locator(xminorLocator)
def ax_rho_mu(ax):
    ax.set_xlim(0,1.5)
    ax.set_ylim(0,20)
    ax.set_xlabel(r'$\rho_B\ $[fm$^{-3}$]',fontsize=50)
    ax.set_ylabel(r'$\mu_i\ $[fm$^{-1}$]',fontsize=50)
    xmajorLocator = MultipleLocator(0.3)  
    ax.xaxis.set_major_locator(xmajorLocator)
    xminorLocator = ticker.MultipleLocator(0.06) 
    ax.xaxis.set_minor_locator(xminorLocator)
    yminorLocator = ticker.MultipleLocator(0.5) 
    ax.yaxis.set_minor_locator(yminorLocator)
def ax_rho_ratio(ax):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(5e-2,2e0)
    ax.set_ylim(1e-2,1e1)
    ax.set_xlabel(r'$\rho_B\ $[fm$^{-3}$]',fontsize=50)
    ax.set_ylabel(r'The$\ $ratio$\ $of$\ $the$\ $density$\ $',fontsize=50)



fig1, ax1 = fig_ax((18,15))
ax_rho_e(ax1)
ax1.plot(RHOB, Etot_DDMEX, '-' , linewidth=3.5, color='blue', label=r'DD-MEX')
ax1.plot(RHOB, Etot_DDME2, '--' , linewidth=3.5, color='red', label=r'DD-ME2')
ax1.legend(fontsize=35, loc=4, bbox_to_anchor=(0.97,0.03))

fig2, ax2 = fig_ax((18,15))
ax_rho_p(ax2)
ax2.plot(RHOB, Ptot_DDMEX, '-' , linewidth=3.5, color='blue', label=r'DD-MEX')
ax2.plot(RHOB, Ptot_DDME2, '--' , linewidth=3.5, color='red', label=r'DD-ME2')
ax2.legend(fontsize=35, loc=4, bbox_to_anchor=(0.97,0.03))

fig3, ax3 = fig_ax((18,15))
ax_rho_mu(ax3)
ax3.plot([0,2],[DDMEX.M_mu,DDMEX.M_mu], '--', linewidth=2.5, color='black')
ax3.plot(RHOB, mu_p_DDMEX, '-.', linewidth=3.5, color='blue', label=r'$\mu_p$(DD-MEX)')
ax3.plot(RHOB, mu_n_DDMEX, '--', linewidth=3.5, color='blue', label=r'$\mu_n$(DD-MEX)')
ax3.plot(RHOB, mu_e_DDMEX, '-',  linewidth=3.5, color='blue', label=r'$\mu_e/\mu_\mu$(DD-MEX)')
ax3.plot(RHOB, mu_p_DDME2, '-.', linewidth=3.5, color='red', label=r'$\mu_p$(DD-ME2)')
ax3.plot(RHOB, mu_n_DDME2, '--', linewidth=3.5, color='red', label=r'$\mu_n$(DD-ME2)')
ax3.plot(RHOB, mu_e_DDME2, '-',  linewidth=3.5, color='red', label=r'$\mu_e/\mu_\mu$(DD-ME2)')
ax3.legend(fontsize=30, loc=2, bbox_to_anchor=(0.03,0.97))

fig4, ax4 = fig_ax((18,15))
ax_rho_ratio(ax4)
l1, = ax4.plot([],[],'-.', linewidth=3.5, color='blue', label='DD-MEX')
l2, = ax4.plot([],[],'--', linewidth=3.5, color='blue', label='DD-ME2')
l3, = ax4.plot(RHOB, rp_DDMEX, '-.', linewidth=3.5, color='blue', label=r'p')
l4, = ax4.plot(RHOB, rn_DDMEX, '-.', linewidth=3.5, color='red',  label=r'n')
l5, = ax4.plot(RHOB, re_DDMEX, '-.', linewidth=3.5, color='orange', label=r'$e^-$')
l6, = ax4.plot(RHOB, rmu_DDMEX,'-.', linewidth=3.5, color='green', label=r'$\mu^-$')
ax4.plot(RHOB, rp_DDME2, '--', linewidth=3.5, color='blue')
ax4.plot(RHOB, rn_DDME2, '--', linewidth=3.5, color='red')
ax4.plot(RHOB, re_DDME2, '--', linewidth=3.5, color='orange')
ax4.plot(RHOB, rmu_DDME2,'--', linewidth=3.5, color='green')
FIG1 = [l1, l2]
FIG2 = [l3, l4, l5, l6]
first_legend = ax4.legend(handles = FIG1, fontsize=25, loc=1, bbox_to_anchor=(0.97,0.97))
ax4.add_artist(first_legend)
ax4.legend(handles = FIG2, fontsize=25, loc=2, bbox_to_anchor=(0.03,0.97))

fig1.savefig('fig5_1a.pdf', bbox_inches='tight')
fig2.savefig('fig5_1b.pdf', bbox_inches='tight')
fig3.savefig('fig5_2a.pdf', bbox_inches='tight')
fig4.savefig('fig5_2b.pdf', bbox_inches='tight')
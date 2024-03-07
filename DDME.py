# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 20:19:26 2024

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
        self.M_P = self.M_N
        self.rho_B0 = 0.152
        self.HC = 197.327 # from MeV to fm^-1
        self.HC4 = self.HC**4
        # 对称因子
        self.S_delta = None
        self.S_gamma = None
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
    
    # 初始化
    def Initialization(self, Value, RHOB):
        self.sigma, self.omega, self.rho = Value 
        # 核物质
        self.Rho_B_P, self.Rho_B_N = (1.0-self.S_delta)*RHOB/2.0, (1.0+self.S_delta)*RHOB/2.0
        self.KF_P = self.FermiMomentum(self.Rho_B_P)
        self.KF_N = self.FermiMomentum(self.Rho_B_N)
        # 代入RHOB
        x = RHOB / self.rho_B0
        self.Coupling(x)
        self.Density()
    # 运动方程
    def EquationsOfMotion(self, InitialValue, RHOB):  
        self.Initialization(InitialValue, RHOB)
        # equation
        eq1 = self.M_sigma**2 * self.sigma - self.coupling_sigma*self.Rho_s
        eq2 = self.M_omega**2 * self.omega - self.coupling_omega*self.Rho_B
        eq3 = self.M_rho**2 * self.rho - 1/2*self.coupling_rho*self.Rho3
        #eq4 = self.Rho_B - RHOB       # Rho_B(N)+Rho_B(P) = RHOB
        #eq5 = self.Rho_B_N/RHOB - 1/2 # 对称核物质
        return np.array([eq1, eq2, eq3])
    
    def FermiMomentum(self, RhoB):
        return (3 * PI2 * RhoB ) ** (1/3)
    # 对称系数
    def Symmetry(self, delta):
        if delta == 0:
            self.S_delta = 0
            self.S_gamma = 4
        elif delta == 1:
            self.S_delta = 1
            self.S_gamma = 2
        else:
            print('非均匀核物质或纯中子物质。')
    
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
        return self.coupling_sigma, self.coupling_omega, self.coupling_rho
    # 密度项
    def CalculateDensity(self, M, KF):
        MS = M - self.coupling_sigma * self.sigma
        EFS = math.sqrt(KF**2 + MS**2)
        Rho_s = 2 /(4*PI2) * MS * (KF*EFS - MS**2*math.log((KF+EFS)/abs(MS)) )
        return MS, Rho_s
    def Density(self):
        self.MS_P, self.Rho_s_P = self.CalculateDensity(self.M_P, self.KF_P)
        self.MS_N, self.Rho_s_N = self.CalculateDensity(self.M_N, self.KF_N)
        # rho_s, rho_s3, rho_B, rho3
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
        return (Ek_P + Ek_N) + US - UW - UR + EW + ER
    def PressureTotal(self, x):
        US = self.M_sigma**2 * self.sigma**2 / 2
        UW = self.M_omega**2 * self.omega**2 / 2
        UR = self.M_rho**2 * self.rho**2 /2       
        Pk_P = self.Pkin(self.MS_P, self.KF_P)
        Pk_N = self.Pkin(self.MS_N, self.KF_N)  
        Rearrangement = self.Sigma_R(x) * self.Rho_B
        return (Pk_P + Pk_N) - US + UW + UR + Rearrangement
    
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
        self.Initialization(Value, RHOB)
        x = RHOB / self.rho_B0
        # 性质
        MSN = self.M_N - self.coupling_sigma * self.sigma
        Etot = self.EnergyTotal()
        Ptot = self.PressureTotal(x)
        EA = Etot/RHOB - self.M_N
        Esym = self.SymmetryEnergy()
        return MSN, Etot*self.HC, Ptot*self.HC, EA*self.HC, Esym*self.HC
    

# 参数
IRHOB = 1001
RHOB = np.linspace(0.001,1.001,IRHOB) 
InitialValues = [0.01, 0.01, 0.01] # 求解初值
DDMEX_parameters = [547.3327, 783, 763, 10.7067, 13.3388, 7.2380, 1.3970, 1.3350,
                    2.0671, 0.4016, 1.3936, 1.0191, 1.6060, 0.4556, 0.6202]

# 初始化模型
DDMEX = DDME_Model()
DDMEX.DDME_parameters(DDMEX_parameters)
DDMEX.Symmetry(delta=0)

# 牛顿迭代法求解运动方程
Newton = Newton_method()
Equations_solve = Newton.NewtonIteration(DDMEX.EquationsOfMotion, InitialValues, RHOB)
sigma_SNM, omega_SNM, rho_SNM = Equations_solve.transpose()


# 计算核物质性质
Properties_SNM = np.zeros((IRHOB,5))
for i in range(IRHOB):
    result = DDMEX.GetProperties(Equations_solve[i], RHOB[i])
    Properties_SNM[i] = result
MSN_SNM, Etot_SNM, Ptot_SNM, EA_SNM, Esym_SNM = Properties_SNM.transpose()
Esls_SNM = DDMEX.SymmetryEnergySlope(Esym_SNM, RHOB)
K_SNM = DDMEX.Imcompressibility(Ptot_SNM, RHOB)

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
SSP = SaturationPointProperties(RHOB, EA_SNM, K_SNM[1], Esym_SNM, Esls_SNM[1], MSN_SNM, 0.152)
print('DD-MEX:', SSP)

# 耦合常数
Coupling_DDMEX = np.zeros((IRHOB,3))
for i in range(IRHOB):
    result = DDMEX.Coupling(RHOB[i]/DDMEX.rho_B0)
    Coupling_DDMEX[i] = result
GS, GW, GR = Coupling_DDMEX.transpose()

### 纯中子物质
DDMEX.Symmetry(delta=1)
Equations_solve = Newton.NewtonIteration(DDMEX.EquationsOfMotion, InitialValues, RHOB)
sigma_PNM, omega_PNM, rho_PNM = Equations_solve.transpose()
Properties_PNM = np.zeros((IRHOB,5))
for i in range(IRHOB):
    result = DDMEX.GetProperties(Equations_solve[i], RHOB[i])
    Properties_PNM[i] = result
MSN_PNM, Etot_PNM, Ptot_PNM, EA_PNM, Esym_PNM = Properties_PNM.transpose()
Esls_PNM = DDMEX.SymmetryEnergySlope(Esym_PNM, RHOB)
K_PNM = DDMEX.Imcompressibility(Ptot_PNM, RHOB)

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
def ax_rho_ea(ax):
    ax.set_xlim(0,0.5)
    ax.set_ylim(-25,150)
    ax.set_xlabel(r'$\rho_B$[fm$^{-3}$]',fontsize=50)
    ax.set_ylabel(r'Binding$\ $energy$\ $per$\ $nucleon$\ $[MeV]',fontsize=50)
    ymajorLocator = MultipleLocator(25)  
    ax.yaxis.set_major_locator(ymajorLocator)
    xminorLocator = ticker.MultipleLocator(0.02) 
    ax.xaxis.set_minor_locator(xminorLocator)
    yminorLocator = ticker.MultipleLocator(5) 
    ax.yaxis.set_minor_locator(yminorLocator)
def ax_rho_p(ax):
    ax.set_yscale('log')
    ax.set_xlim(0,0.5)
    ax.set_ylim(1e-2,1e3)
    ax.set_xlabel(r'$\rho_B\ $[fm$^{-3}$]',fontsize=50)
    ax.set_ylabel(r'P$\ $[MeV]',fontsize=50)
    xminorLocator = ticker.MultipleLocator(0.02) 
    ax.xaxis.set_minor_locator(xminorLocator)
def ax_rho_esym(ax):
    ax.set_xlim(0,0.5)
    ax.set_ylim(0,80)
    ax.set_xlabel(r'$\rho_B$[fm$^{-3}$]',fontsize=50)
    ax.set_ylabel(r'E$_{sym}\ $[MeV]',fontsize=50)
    xminorLocator = ticker.MultipleLocator(0.02) 
    ax.xaxis.set_minor_locator(xminorLocator)
    yminorLocator = ticker.MultipleLocator(2) 
    ax.yaxis.set_minor_locator(yminorLocator)
def ax_rho_esls(ax):
    ax.set_xlim(0,0.5)
    ax.set_ylim(0,80)
    ax.set_xlabel(r'$\rho_B$[fm$^{-3}$]',fontsize=50)
    ax.set_ylabel(r'The$\ $slope$\ $of$\ $symmetry$\ $energy',fontsize=50)
    xminorLocator = ticker.MultipleLocator(0.02) 
    ax.xaxis.set_minor_locator(xminorLocator)
    yminorLocator = ticker.MultipleLocator(2) 
    ax.yaxis.set_minor_locator(yminorLocator)
def ax_rho_gamma(ax, particle):
    ax.set_xlim(0,1)
    ax.set_xlabel(r'$\rho_B$[fm$^{-3}$]',fontsize=50)
    if particle == 'sigma':
        ax.set_ylim(7,14)
        ax.set_ylabel(r'$\Gamma_\sigma(\rho_B)$',fontsize=50)
        yminorLocator = ticker.MultipleLocator(0.2) 
        ax.yaxis.set_minor_locator(yminorLocator)   
    elif particle == 'omega':
        ax.set_ylim(9,17)
        ax.set_ylabel(r'$\Gamma_\omega(\rho_B)$',fontsize=50)  
        yminorLocator = ticker.MultipleLocator(0.2) 
        ax.yaxis.set_minor_locator(yminorLocator)   
    elif particle == 'rho':
        ax.set_ylim(0,16)
        ax.set_ylabel(r'$\Gamma_\rho(\rho_B)$',fontsize=50)   
        yminorLocator = ticker.MultipleLocator(0.4) 
        ax.yaxis.set_minor_locator(yminorLocator)   
    xminorLocator = ticker.MultipleLocator(0.04) 
    ax.xaxis.set_minor_locator(xminorLocator)
 

fig1, ax1 = fig_ax((18,15))
ax_rho_ea(ax1)
ax1.plot(RHOB, EA_SNM, '-' , linewidth=3.5, color='blue', label=r'DD-MEX($\delta$=0)')
ax1.plot(RHOB, EA_PNM, '--', linewidth=3.5, color='blue', label=r'DD-MEX($\delta$=1)')
ax1.legend(fontsize=35, loc=2, bbox_to_anchor=(0.03,0.97))

fig2, ax2 = fig_ax((18,15))
ax_rho_p(ax2)
ax2.plot(RHOB, Ptot_SNM, '-' , linewidth=3.5, color='blue', label=r'DD-MEX($\delta$=0)')
ax2.plot(RHOB, Ptot_PNM, '--', linewidth=3.5, color='blue', label=r'DD-MEX($\delta$=1)')
ax2.legend(fontsize=35, loc=4, bbox_to_anchor=(0.97,0.03))


####################################################################################
### DD-ME2
DDME2_parameters = [550.1238, 783, 763, 10.5396, 13.0189, 7.3672, 1.3881, 1.0943,
                    1.7057, 0.4421, 1.3892, 0.9240, 1.4620, 0.4775, 0.5647]
DDME2 = DDME_Model()
DDME2.DDME_parameters(DDME2_parameters)
# 对称核物质
DDME2.Symmetry(delta=0)
Equations_solve = Newton.NewtonIteration(DDME2.EquationsOfMotion, InitialValues, RHOB)
Properties_SNM2 = np.zeros((IRHOB,5))
for i in range(IRHOB):
    result = DDME2.GetProperties(Equations_solve[i], RHOB[i])
    Properties_SNM2[i] = result
MSN_SNM2, Etot_SNM2, Ptot_SNM2, EA_SNM2, Esym_SNM2 = Properties_SNM2.transpose()
Esls_SNM2 = DDME2.SymmetryEnergySlope(Esym_SNM2, RHOB)
K_SNM2 = DDME2.Imcompressibility(Ptot_SNM2, RHOB)
SSP2 = SaturationPointProperties(RHOB, EA_SNM2, K_SNM2[1], Esym_SNM2, Esls_SNM2[1], MSN_SNM2, 0.152)
print('DD-ME2:', SSP2)
#
Coupling_DDME2 = np.zeros((IRHOB,3))
for i in range(IRHOB):
    result = DDME2.Coupling(RHOB[i]/DDME2.rho_B0)
    Coupling_DDME2[i] = result
GS2, GW2, GR2 = Coupling_DDME2.transpose()
# 纯中子物质
DDME2.Symmetry(delta=1)
Equations_solve = Newton.NewtonIteration(DDME2.EquationsOfMotion, InitialValues, RHOB)
Properties_PNM2 = np.zeros((IRHOB,5))
for i in range(IRHOB):
    result = DDME2.GetProperties(Equations_solve[i], RHOB[i])
    Properties_PNM2[i] = result
MSN_PNM2, Etot_PNM2, Ptot_PNM2, EA_PNM2, Esym_PNM2 = Properties_PNM2.transpose()
Esls_PNM2 = DDME2.SymmetryEnergySlope(Esym_PNM2, RHOB)
K_PNM2 = DDME2.Imcompressibility(Ptot_PNM2, RHOB)



fig3, ax3 = fig_ax((18,15))
ax_rho_ea(ax3)
ax3.plot(RHOB, EA_SNM, '-' , linewidth=3.5, color='blue', label=r'DD-MEX($\delta$=0)')
ax3.plot(RHOB, EA_PNM, '--', linewidth=3.5, color='blue', label=r'DD-MEX($\delta$=1)')
ax3.plot(RHOB, EA_SNM2, '-' , linewidth=3.5, color='red', label=r'DD-ME2($\delta$=0)')
ax3.plot(RHOB, EA_PNM2, '--', linewidth=3.5, color='red', label=r'DD-ME2($\delta$=1)')
ax3.legend(fontsize=35, loc=2, bbox_to_anchor=(0.03,0.97))

fig4, ax4 = fig_ax((18,15))
ax_rho_p(ax4)
ax4.plot(RHOB, Ptot_SNM, '-' , linewidth=3.5, color='blue', label=r'DD-MEX($\delta$=0)')
ax4.plot(RHOB, Ptot_PNM, '--', linewidth=3.5, color='blue', label=r'DD-MEX($\delta$=1)')
ax4.plot(RHOB, Ptot_SNM2, '-' , linewidth=3.5, color='red', label=r'DD-ME2($\delta$=0)')
ax4.plot(RHOB, Ptot_PNM2, '--', linewidth=3.5, color='red', label=r'DD-ME2($\delta$=1)')
ax4.legend(fontsize=35, loc=4, bbox_to_anchor=(0.97,0.03))


fig5, ax5 = fig_ax((18,15))
ax_rho_esym(ax5)
ax5.plot(RHOB, Esym_SNM, '-' , linewidth=3.5, color='blue', label=r'DD-MEX')
ax5.plot(RHOB, Esym_SNM2, '--', linewidth=3.5, color='red', label=r'DD-ME2')
ax5.legend(fontsize=35, loc=4, bbox_to_anchor=(0.97,0.03))

fig6, ax6 = fig_ax((18,15))
ax_rho_esls(ax6)
ax6.plot(Esls_SNM[0], Esls_SNM[1],  '-' , linewidth=3.5, color='blue', label=r'DD-MEX')
ax6.plot(Esls_SNM2[0], Esls_SNM2[1], '--', linewidth=3.5, color='red', label=r'DD-ME2')
ax6.legend(fontsize=35, loc=4, bbox_to_anchor=(0.97,0.03))

fig7, ax7 = fig_ax((18,15))
ax_rho_gamma(ax7, 'sigma')
ax7.plot(RHOB, GS, '-' , linewidth=3.5, color='blue', label=r'DD-MEX')
ax7.plot(RHOB, GS2, '--', linewidth=3.5, color='red', label=r'DD-ME2')
ax7.legend(fontsize=35, loc=1, bbox_to_anchor=(0.97,0.97))

fig8, ax8 = fig_ax((18,15))
ax_rho_gamma(ax8, 'omega')
ax8.plot(RHOB, GW, '-' , linewidth=3.5, color='blue', label=r'DD-MEX')
ax8.plot(RHOB, GW2, '--', linewidth=3.5, color='red', label=r'DD-ME2')
ax8.legend(fontsize=35, loc=1, bbox_to_anchor=(0.97,0.97))

fig9, ax9 = fig_ax((18,15))
ax_rho_gamma(ax9, 'rho')
ax9.plot(RHOB, GR, '-' , linewidth=3.5, color='blue', label=r'DD-MEX')
ax9.plot(RHOB, GR2, '--', linewidth=3.5, color='red', label=r'DD-ME2')
ax9.legend(fontsize=35, loc=1, bbox_to_anchor=(0.97,0.97))

fig1.savefig('fig4_1a.pdf', bbox_inches='tight')
fig2.savefig('fig4_1b.pdf', bbox_inches='tight')
fig3.savefig('fig4_2a.pdf', bbox_inches='tight')
fig4.savefig('fig4_2b.pdf', bbox_inches='tight')
fig5.savefig('fig4_3a.pdf', bbox_inches='tight')
fig6.savefig('fig4_3b.pdf', bbox_inches='tight')
fig7.savefig('fig4_4a.pdf', bbox_inches='tight')
fig8.savefig('fig4_4b.pdf', bbox_inches='tight')
fig9.savefig('fig4_4c.pdf', bbox_inches='tight')
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:40:08 2024

@author: erika
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

PI2 = math.pi**2

class Newton_method():
    def __init__(self, max_iter=100, tol=1e-6, epsilon=1e-10):
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
      
class Walecka_Model:
    def __init__(self):
        # 定义模型参数
        self.M_N = 939.0  # 质子和中子的质量（单位：MeV/c^2）
        self.M_P = self.M_N
        self.rho_B0 = 0.152
        self.HC = 197.327 # from MeV to fm^-1
    def Parameters(self, parameters):
        m_sigma, m_omega, g_sigma, g_omega = parameters
        self.M_sigma = m_sigma  # 介子质量
        self.M_omega = m_omega
        self.g_sigma = g_sigma  # 核子-σ介子耦合常数
        self.g_omega = g_omega  # σ介子自相互作用常数
        # 单位转换
        self.M_N /= self.HC
        self.M_P /= self.HC
        self.M_sigma /= self.HC
        self.M_omega /= self.HC

    # 运动方程
    def EquationsOfMotion(self, IV, RHOB):  
        self.phi0, self.V0, self.KF_P, self.KF_N = IV # 初值
        # 代入RHOB
        self.Density()
        # equation
        eq1 = self.phi0 * self.M_sigma**2 - self.g_sigma  * self.Rho_s
        eq2 = self.V0 * self.M_omega**2   - self.g_omega  * self.Rho_B
        eq3 = self.Rho_B - RHOB       # Rho_B(N)+Rho_B(P) = RHOB
        eq4 = self.Rho_B_N/RHOB - (1-self.S_delta)/2 # 对称核物质 or 纯中子物质
        return np.array([eq1, eq2, eq3, eq4])       
        
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
    # 密度项
    def CalculateDensity(self, M, KF):
        MS = M - self.g_sigma * self.phi0
        EFS = math.sqrt(KF**2 + MS**2)
        Rho_s = 2/(4*PI2) * MS * (KF*EFS - MS**2*math.log((KF+EFS)/abs(MS)) )
        Rho_B = 2/(6*PI2) * KF**3
        return MS, Rho_s, Rho_B
    def Density(self):
        self.MS_P, self.Rho_s_P, self.Rho_B_P = self.CalculateDensity(self.M_P, self.KF_P)
        self.MS_N, self.Rho_s_N, self.Rho_B_N = self.CalculateDensity(self.M_N, self.KF_N)
        # rho_s, rho_s3, rho_B, rho3
        self.Rho_s = self.Rho_s_P + self.Rho_s_N
        self.Rho_s3 = self.Rho_s_P - self.Rho_s_N
        self.Rho_B = self.Rho_B_P + self.Rho_B_N
        self.Rho3 = self.Rho_B_P - self.Rho_B_N
    
    # 压强和能量密度
    def Ekin(self, MS, KF): # M*, Kf
        EFS = math.sqrt(KF**2 + MS**2)
        return (2 / 16.0 / PI2) * ((2.0 * KF**3 +  MS**2 * KF) * EFS -
                                     MS**4 * math.log((KF + EFS) / abs(MS)))
    def Pkin(self, MS, KF):
        EFS = math.sqrt(KF**2 + MS**2)
        return (2 / 48.0 / PI2) * ((2.0 * KF**3 - 3.0 * MS**2 * KF) * EFS + 
                                     3.0 * MS**4 * math.log((KF + EFS) / abs(MS)))
    def E(self):
        EK_P = self.Ekin(self.MS_P, self.KF_P)
        EK_N = self.Ekin(self.MS_N, self.KF_N)
        UW = 1/2 * self.M_omega**2 * self.V0**2
        US = 1/2 * self.M_sigma**2 * self.phi0**2
        return (EK_P+EK_N) + UW + US
    def P(self):
        PK_P = self.Pkin(self.MS_P, self.KF_P)
        PK_N = self.Pkin(self.MS_N, self.KF_N)
        UW = 1/2 * self.M_omega**2 * self.V0**2
        US = 1/2 * self.M_sigma**2 * self.phi0**2
        return (PK_P+PK_N) + UW - US 
    def GetProperties(self, Value, RHOB):
        self.phi0, self.V0, self.KF_P, self.KF_N = Value # 初值
        self.Density()
        # 性质
        Etot = self.E()
        Ptot = self.P()
        EA = Etot/RHOB - self.M_N
        return Etot*self.HC, Ptot*self.HC, EA*self.HC

# 
Walecka_parameters = [550.0, 783.0, 10.28, 12.613]
Value = [0.01,0.01,0.01,0.01]
rhob = np.linspace(0.001, 1.001, 1001)  # 密度范围（单位：fm^-3）
# 初始化模型
WM = Walecka_Model()
Newton = Newton_method()
# 牛顿迭代法求解运动方程
WM.Parameters(Walecka_parameters)
WM.Symmetry(0)
Equations_solve = Newton.NewtonIteration(WM.EquationsOfMotion, Value, rhob)
# 计算核物质性质
Properties_SNM = np.zeros((1001,3))
for i in range(1001):
    result = WM.GetProperties(Equations_solve[i], rhob[i])
    Properties_SNM[i] = result
Etot_SNM, Ptot_SNM, EA_SNM = Properties_SNM.transpose()


# 纯中子物质
WM.Symmetry(1)
Equations_solve = Newton.NewtonIteration(WM.EquationsOfMotion, Value, rhob)
# 计算核物质性质
Properties_PNM = np.zeros((1001,3))
for i in range(1001):
    result = WM.GetProperties(Equations_solve[i], rhob[i])
    Properties_PNM[i] = result
Etot_PNM, Ptot_PNM, EA_PNM = Properties_PNM.transpose()


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
    ax.set_xlim(0,0.4)
    ax.set_ylim(-20,40)
    ax.set_xlabel(r'$\rho_B$[fm$^{-3}$]',fontsize=50)
    ax.set_ylabel(r'Binding$\ $energy$\ $per$\ $nucleon$\ $[MeV]',fontsize=50)
    xmajorLocator = MultipleLocator(0.1)  
    ax.xaxis.set_major_locator(xmajorLocator)
    ymajorLocator = MultipleLocator(10)  
    ax.yaxis.set_major_locator(ymajorLocator)
    xminorLocator = ticker.MultipleLocator(0.02) 
    ax.xaxis.set_minor_locator(xminorLocator)
    yminorLocator = ticker.MultipleLocator(2) 
    ax.yaxis.set_minor_locator(yminorLocator)
def ax_rho_p(ax):
    ax.set_yscale('log')
    ax.set_xlim(0,0.5)
    ax.set_ylim(5e-2,1e3)
    ax.set_xlabel(r'$\rho_B\ $[fm$^{-3}$]',fontsize=50)
    ax.set_ylabel(r'P$\ $[MeV]',fontsize=50)
    xminorLocator = ticker.MultipleLocator(0.02) 
    ax.xaxis.set_minor_locator(xminorLocator)

fig1, ax1 = fig_ax((18,15))
ax_rho_ea(ax1)
ax1.plot(rhob, EA_SNM, '-' , linewidth=3.5, color='blue', label=r'Walecka($\delta$=0)')
ax1.plot(rhob, EA_PNM, '--', linewidth=3.5, color='blue', label=r'Walecka($\delta$=1)')
ax1.legend(fontsize=35, loc=2, bbox_to_anchor=(0.03,0.97))

fig2, ax2 = fig_ax((18,15))
ax_rho_p(ax2)
ax2.plot(rhob, Ptot_SNM, '-' , linewidth=3.5, color='blue', label=r'Walecka($\delta$=0)')
ax2.plot(rhob, Ptot_PNM, '--', linewidth=3.5, color='blue', label=r'Walecka($\delta$=1)')
ax2.legend(fontsize=35, loc=4, bbox_to_anchor=(0.97,0.03))

fig1.savefig('fig3_1a.pdf', bbox_inches='tight')
fig2.savefig('fig3_1b.pdf', bbox_inches='tight')

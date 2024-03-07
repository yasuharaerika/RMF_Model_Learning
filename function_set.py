# -*- coding: utf-8 -*-
"""
Created on Sat May 28 21:26:11 2022

@author: erika
"""

import numpy as np
import math

# basic parameters
#
# Energy density
epsilon_0 = 2.7e17 # kg/m**3
epsilon_0 = epsilon_0 / 1.79e-30 * (1e-15)**3 # Mev/fm**3
# The confidence coefficient, p=95%.
co_co = 1.96
# Value added in the iteration of a differential equation.
delta = 0.001

# Shell
Q = np.array([0.184,9.34375e-5,4.1725e-8,1.44875e-11,6.3125e-25])
Q0 = np.array([15.54878,0.66635,76.70887,0.24734])
Q1 = np.array([0.00873,103.17338,1/0.38527,7.34979,1/0.01211])
Q2 = np.array([0.00015,0.00202,344827.5,0.10851,7692.3076])
Q3 = np.array([0.0000051,0.2373e10,0.00014,0.4021e8])
Q4 = np.array([31.93753,10.82611,1.29312,0.08014,0.00242,0.000028])

# Generate M-R-L
class Generate_MRL():
    def __init__(self, optimize=True):
        # 常数
        self.steps, self.epsilon_0, self.beta = 0.01, 1.0, 11.2e-6
        self.RR = np.arange(1e-5,25,1e-5) # R range
        self.M_0, self.R_0, self.y_0 = 1e-10, 1.474, 2
        # 拟合状态
        self.basic_pe, self.basic_eos = False, False
        self.fit_pe, self.fit_eos = False, False
        self.is_fit = False
        # P-E拟合型EoS
        self.basic_P, self.basic_E = None, None
        self.P, self.E = None, None
        self.p0, self.shell_p0 = None, None
        # 直接输入型EoS
        self.basic_EoS, self.EoS = None, None
        self.optimize = optimize
    # 默认壳层EOS
    def Default_Shell(self, p): # MeV
        lg = math.log10(p)
        k1,g1,k2,g2 = Q0
        A1,B1,C1,D1,E1 = Q1
        A2,B2,C2,D2,E2 = Q2
        B3,C3,D3,E3 = Q3
        A4,B4,C4,D4,E4,F4 = Q4
        if p > Q[0]:
            e = k1*p**g1 + k2*p**g2 
        elif p >= Q[1] and p <= Q[0]:
            e = A1 + B1*(1-math.exp(-p*C1)) + D1*(1-math.exp(-p*E1))
        elif p >= Q[2] and p < Q[1]:
            e = A2 + B2*(1-math.exp(-p*C2)) + D2*(1-math.exp(-p*E2))
        elif p >= Q[3] and p < Q[2]:
            e =      B3*(1-math.exp(-p*C3)) + D3*(1-math.exp(-p*E3))
        elif p >= Q[4] and p < Q[3]:
            e = 10**(A4 + B4*lg + C4*lg**2 + D4*lg**3 + E4*lg**4 + F4*lg**5)
        return e
    # 壳层输入EOS
    def Fit_Shell_PE(self, P, E, p0):
        self.basic_P, self.basic_E = np.asarray(P), np.asarray(E)
        self.shell_p0 = p0
        self.basic_pe = True
    def Fit_Shell_EoS(self, EoS, p0):
        self.basic_EoS = EoS
        self.shell_p0 = p0
        self.basic_eos = True
    # 拟合部分EOS输入参数
    def Fit_PE(self, P, E, p0):
        self.P, self.E = np.asarray(P), np.asarray(E)
        self.p0 = p0
        self.is_fit, self.fit_pe = True, True
    def Fit_EoS(self, EoS, p0):
        self.EoS = EoS
        self.p0 = p0
        self.is_fit, self.fit_eos = True, True
    # EOS计算
    def Get_EoS(self, P, E, p):
        low = 0
        high = len(P)-1
        while (high-low) > 1:
            mid = (high+low)//2
            if P[mid] >= p:
                high = mid
            else:
                low = mid
        dedp = (E[low+1]-E[low])/(P[low+1]-P[low])
        return E[low] + dedp*(p-P[low])
    # 求出对应Epsilon    
    def Get_E(self, p):
        # 无衔接   
        if not self.is_fit:
            if p <= self.shell_p0:
                e = self.Default_Shell(p)
            elif self.basic_pe:
                e = self.Get_EoS(self.basic_P, self.basic_E, p)
            elif self.basic_eos:
                e = self.basic_EoS(p)
        else:
            if p <= self.p0:                     # 衔接壳层
                if p <= self.shell_p0:
                    e = self.Default_Shell(p)
                elif self.basic_pe:
                    e = self.Get_EoS(self.basic_P, self.basic_E, p)
                elif self.basic_eos:
                    e = self.basic_EoS(p)
            else:                               # 衔接数据EOS
                if self.fit_pe:
                    e = self.Get_EoS(self.P, self.E, p)
                elif self.fit_eos:
                    e = self.EoS(p)
        return e
    # 4-th order Runge Kutta method 
    def F_pm(self, r, p, m, e):
        # dpdr
        #print('m =',m,'r =',r)
        dpdr = -self.R_0*e*m/r**2 * (1+p/e) * (1+self.beta*r**3*p/m) \
                / (1-2*m*self.R_0/r)   
        # dmdr
        dmdr = self.beta * r**2 * e   
        return dpdr,dmdr
    def runge_kutta_pm(self, t, x, y, h, e):
        hh = h/2
        k1,j1 = self.F_pm(t   ,x      ,y      ,e)
        k2,j2 = self.F_pm(t+hh,x+hh*k1,y+hh*j1,e)
        k3,j3 = self.F_pm(t+hh,x+hh*k2,y+hh*j2,e)
        k4,j4 = self.F_pm(t+ h,x+h*k3 ,y+h*j3 ,e)      
        dX = (k1+2*k2+2*k3+k4)*h/6
        dY = (j1+2*j2+2*j3+j4)*h/6
        return dX,dY
    def F_pmy(self, r, p, m, e, y_r, dpde):
        # dpdr
        dpdr = -self.R_0*e*m/r**2 * (1+p/e) * (1+self.beta*r**3*p/m)  \
                / (1-2*m*self.R_0/r)    
        # dmdr
        dmdr = self.beta * r**2 * e   
        # dydr
        Fr = (1-r**2*self.beta*self.R_0*(e-p)) / (1-2*self.R_0*m/r)
        q1 = self.beta*self.R_0* (5*e+9*p+(e+p)/dpde) / (1-2*self.R_0*m/r)
        q2 = 6/(r**2-2*r*self.R_0*m)
        q3 = 4*self.R_0**2*m**2/r**4 * (1+self.beta*r**3*p/m)**2 / (1-2*m*self.R_0/r)
        Qr = q1 - q2 - q3
        dydr = -(y_r**2+y_r*Fr+r**2*Qr)/r    
        return dpdr,dmdr,dydr
    def runge_kutta_pmy(self, t, x, y, h, e, y_r, dpde):
        hh = h/2
        k1,j1,l1 = self.F_pmy(t   ,x      ,y      ,e,y_r      ,dpde)
        k2,j2,l2 = self.F_pmy(t+hh,x+hh*k1,y+hh*j1,e,y_r+hh*l1,dpde)
        k3,j3,l3 = self.F_pmy(t+hh,x+hh*k2,y+hh*j2,e,y_r+hh*l2,dpde)
        k4,j4,l4 = self.F_pmy(t+ h,x+h*k3 ,y+h*j3 ,e,y_r+h*l3 ,dpde)
        dX = (k1+2*k2+2*k3+k4)*h/6
        dY = (j1+2*j2+2*j3+j4)*h/6
        dZ = (l1+2*l2+2*l3+l4)*h/6
        return dX,dY,dZ
    # 迭代函数
    def Get_cs(self, p):
        delta = 0.001
        cs = p*delta / (self.Get_E(p*(1+delta)) - self.Get_E(p))
        return cs
    def move(self, T, steps):
        m,p,r = T
        # TOV方程
        e = self.Get_E(p)
        dp,dm = self.runge_kutta_pm(r,p,m,steps,e)
        dr = steps
        return [m+dm, p+dp, r+dr]
    def MOVE(self, T, steps):
        m,p,r,y = T
        # TOV方程
        e = self.Get_E(p)
        dpde = self.Get_cs(p)
        dp,dm,dy = self.runge_kutta_pmy(r,p,m,steps,e,y,dpde)
        dr = steps
        return [m+dm, p+dp, r+dr, y+dy]
    # MR关系函数
    def MR(self, T0, RR, n):
        T = T0    
        for r in RR:
            if n == 0:
                t = self.move(T,self.steps)
            else:
                t = self.MOVE(T,self.steps)
            if t[1] < 1e-24:  # 压强小于某个值
                return T
            else:
                T = t
    # k2
    def get_k2(self, C, y_R):
        a1 = 8*C**5/5 * (1-2*C)**2
        a2  = 2 + 2*C*(y_R-1) - y_R
        b1 = 6 - 3*y_R + 3*C*(5*y_R-8)
        b2 = 13 - 11*y_R + C*(3*y_R-2) + 2*C**2*(1+y_R)
        b3 = 2 - y_R + 2*C*(y_R-1)
        b4 = math.log(1-2*C)
        k2 = a1*a2 / (2*C*b1 + 4*C**3*b2 + 3*(1-2*C)**2 * b3*b4)
        return k2
    # MR
    def get_MR(self, P):
        m = []
        r = []
        for p in P:
            #print(p)
            T0 = [self.M_0, p, self.RR[0]]
            mr = self.MR(T0, self.RR, 0)
            m.append(mr[0])
            r.append(mr[2])
        M = np.array(m)
        R = np.array(r)
        return M,R
    # MRL
    def get_MRL(self, P):
        m = []
        r = []
        y = []
        Lam = []
        for p in P:
            T0 = [self.M_0, p, self.RR[0], self.y_0]
            mr = self.MR(T0, self.RR, 1)
            m.append(mr[0])
            r.append(mr[2])
            y.append(mr[3])
            C = self.R_0*mr[0]/mr[2]
            lam = 2/3 * self.get_k2(C,mr[3]) * C**(-5)
            Lam.append(lam)
        M = np.array(m)
        R = np.array(r)
        L = np.array(Lam)
        return M,R,L

# 生成训练数据点
class data_point():
    
    def __init__(self, optimize=True):
        # 常数
        self.e_0 = epsilon_0
        # 
        self.is_fit = False
        self.EOS_NUM = 0
        self.P, self. E = [], []
        self.optimize = optimize
    # 输入EOS数据点
    def fit(self, P, E):
        self.P.append(P)
        self.E.append(E)
        self.EOS_NUM += 1
        self.is_fit = True
    # 生成Phi区间
    def fit_phi(self, train_p):
        if not self.is_fit:
            print('EOS is not fit.')
        else:
            self.phi = []
            for i in range(len(self.P)):
                phi_i = np.array([self.PHI(p,self.P[i],self.E[i]) for p in train_p])
                self.phi.append(phi_i)
            return np.mean(self.phi,axis=0), np.std(self.phi,axis=0)
    # search
    def search_y(self, x, X, Y):
        low = 0
        high = len(X)-1
        while (high-low) > 1:
            mid = (high+low)//2
            if X[mid] >= x:
                high = mid
            else:
                low = mid
        dydx = (Y[high]-Y[low])/(X[high]-X[low])
        return Y[low] + dydx*(x-X[low])
    def search_dydx(self, x, X, Y):
        dx = x * delta
        dy = self.search_y(x+dx, X, Y) - self.search_y(x, X, Y)
        return dy/dx
    # 根据输入epsilon，拟合输入EOS的对应P的范围
    def get_p(self, fit_e):
        if not self.is_fit:
            print('EOS is not fit.')
        else:
            train_p = []
            for i in range(len(fit_e)):
                p = []
                for j in range(self.EOS_NUM):
                    p.append(self.search_y(fit_e[i], self.E[j], self.P[j]))
                train_p.append([np.mean(p), np.std(p)])
            MU,STD = np.split(np.array(train_p),2,axis=1)
            train_P = [MU.reshape(1,-1)[0],STD.reshape(1,-1)[0]]
            return np.array(train_P)
    # 根据输入p，拟合输入EOS的对应epsilon的范围
    def get_e(self, fit_p):
        if not self.is_fit:
            print('EOS is not fit.')
        else:
            train_e = []
            for i in range(len(fit_p)):
                e = []
                for j in range(self.EOS_NUM):
                    e.append(self.search_y(fit_p[i], self.P[j], self.E[j]))
                train_e.append([np.mean(e), np.std(e)])
            MU,STD = np.split(np.array(train_e),2,axis=1)
            train_E = [MU.reshape(1,-1)[0],STD.reshape(1,-1)[0]]
            return np.array(train_E)
    # 根据输入epsilo，拟合输入EOS的声速范围
    def get_cs(self, fit_e):
        if not self.is_fit:
            print('EOS is not fit.')
        else:
            train_cs = []
            for i in range(len(fit_e)):
                cs = []
                for j in range(self.EOS_NUM):
                    cs.append(self.search_dydx(fit_e[i], self.E[j], self.P[j]))
                train_cs.append([np.mean(cs),np.std(cs)])
            MU,STD = np.split(np.array(train_cs),2,axis=1)
            train_CS = [MU.reshape(1,-1)[0],STD.reshape(1,-1)[0]]
            return np.array(train_CS)
    # 生成拟合参数p-epsilon / 以临近两点作为一点拟合数据
    def get_data_point(self, data_p, phi_0, n):
        train_phi = [phi_0]
        for i in range(len(data_p)-1):
            # p
            if n == 0:
                mu, std = self.fit_phi(np.array([data_p[i+1]]))
                phi_up = (mu + co_co*std)[0]
                phi_down = (mu - co_co*std)[0]
            else:
                mu = self.PHI(data_p[i+1], self.P[0], self.E[0])
                std = 0.3
                phi_up = mu + std
                phi_down = mu - std
            # 无限制
            phi = np.random.uniform(phi_down,phi_up)
            # 加强声速限制
            #phi = np.random.uniform(phi_down,max(train_phi[-1],phi_up))[0]
            train_phi.append(phi)
        return np.array(train_phi)
    # 生成训练参数phi-lnp
    def PHI(self, p, P, E): # 输入 ln(p),[P,E]
        X,Y = np.log(P), np.log(E)
        low = 0
        high = len(P)-1
        while (high-low) > 1:
            mid = (high+low)//2
            if X[mid] >= p:
                high = mid
            else:
                low = mid
        i = low
        dedp = (Y[i+1]-Y[i])/(X[i+1]-X[i])
        loge = Y[i] + (Y[i+1]-Y[i])/(X[i+1]-X[i])*(p-X[i])
        return np.log(dedp*np.exp(loge)/np.exp(p)-1)
    def Phi(self, p, P, E): # 输入 p,[P,E]
        low = 0
        high = len(P)-1
        while (high-low) > 1:
            mid = (high+low)//2
            if P[mid] >= p:
                high = mid
            else:
                low = mid
        dedp = (E[low+1]-E[low])/(P[low+1]-P[low])
        return np.log(dedp-1)
    # 由声速直接计算
    def phi(self, cs):
        return np.log(1/cs - 1)
    # phi-p 积分为 epsilon-p
    def int_phi(self, phi, p, e0):
        L = len(p)
        e = [e0]
        for i in range(L-1):
            dedp = 1 + math.exp(phi[i+1])
            e.append(e[-1] + (p[i+1]-p[i])*dedp)
        return np.array(e).reshape(1,-1)[0]


# 修复lambda计算中的误差
def fix_l(M,LAMBDA):
    if max(M) <= 0.95 or min(M) >= 2.01:
        return M,LAMBDA
    m = M[M>0.95]
    num = len(M)-len(m)
    m = m[m<2.01]
    Lambda = LAMBDA[np.argwhere(M>0.95)].reshape(1,-1)[0]
    Lambda = Lambda[np.argwhere(m<2.01)].reshape(1,-1)[0]
    L = [Lambda[0]]
    test = 0 # 基准点值，下一个lamda值比基准lamda值小时相差距离
    for i in range(len(Lambda)-2):
        if Lambda[i+1] <= Lambda[i] and test == 0:
            L.append(Lambda[i+1])
        elif Lambda[i+1] <= Lambda[i-test] and test != 0:
            L.append(Lambda[i+1])
            test = 0
        else:
            if Lambda[i] > 100:
                error = 0.05
            else:
                error = 0.2 
            test += 1
            k = 2
            while Lambda[i+k]-L[-1] > L[-1]*error and i+k != len(Lambda)-1:
                k += 1
            l = L[-1] - (L[-1]-LAMBDA[i+k+num])/k
            L.append(l)
    L.append(Lambda[-1])
    return m,np.array(L)

# 生成潮汐形变阴影区间
def get_y(M,x,j):
    x1,x2 = M[0][j],M[0][j+1]
    y1,y2 = M[1][j],M[1][j+1]
    k = (y2-y1)/(x2-x1)
    return k*(x-x1) + y1

def new_ml(m1,l1,m2,l2):
    l1 = l1[np.argwhere(m1>=0.98)].reshape(1,-1)[0]
    m1 = m1[m1>=0.98]
    l2 = l2[np.argwhere(m2>=0.98)].reshape(1,-1)[0]
    m2 = m2[m2>=0.98]
    
    if max(l1) > max(l2):
        MAX = [m1,l1]
        MIN = [m2,l2]
    else:
        MAX = [m2,l2]
        MIN = [m1,l1]
    up_lim = min(2, min(max(m1),max(m2)))
    x = np.linspace(1,up_lim,10001)
    yu = []
    yd = []
    for i in range(len(x)):
        for j in range(len(MAX[1])-1):
            if MAX[0][j] <= x[i] and x[i] <= MAX[0][j+1]:
                yu.append(get_y(MAX,x[i],j))
                break
        for j in range(len(MIN[1])-1):
            if MIN[0][j] <= x[i] and x[i] <= MIN[0][j+1]:
                yd.append(get_y(MIN,x[i],j))
                break
    return x,np.array(yu),np.array(yd)


# 生成M-R
def MEAN(R,M):     # 对于数据量较大但迭代间距较小而导致同一个半径对应多个M进行平均修正
    r = [R[0]]
    for i in range(1,len(R)):
        if R[i] != r[-1]:
            r.append(R[i])
    m = []
    num,j,SUM = 0,0,0
    for i in range(len(R)-1):
        if R[i] == r[j]:
            SUM += M[i]
            num += 1
        elif R[i] != r[j]:
            m.append(SUM/num)
            j += 1
            SUM,num = M[i],1
        if i+1 == len(R)-1:
            if R[i+1] == r[j]:
                SUM += M[i]
                num += 1
                m.append(SUM/num)
            elif R[i+1] != r[j]:
                m.append(SUM/num)
                m.append(M[i+1])
    return np.array(m),np.array(r)


# 生成M-R阴影区间
def mr(m,r):
    M_max = np.argwhere(m==max(m))[0][0]
    M = m[:M_max+1]
    R = r[:M_max+1]
    return R,M

def get_x(M,y,j):
    x1,x2 = M[0][j],M[0][j+1]
    y1,y2 = M[1][j],M[1][j+1]
    k = (x2-x1)/(y2-y1)
    return k*(y-y1) + x1

# 上下限曲线各自到最大质量处
def new_mr(m1,r1,m2,r2):
    r1,m1 = mr(m1,r1)
    r2,m2 = mr(m2,r2)
    if max(m1) > max(m2):
        MAX = [r1,m1]
        MIN = [r2,m2]
    else:
        MAX = [r2,m2]
        MIN = [r1,m1]
    min_m = max(0.1,min(MIN[1]),min(MAX[1]))
    y = np.linspace(min_m,max(MAX[1]),10000)
    xu = []
    xd = []
    for i in range(len(y)):
        for j in range(len(MAX[1])-1):
            if MAX[1][j] <= y[i] and y[i] <= MAX[1][j+1]:
                xu.append(get_x(MAX,y[i],j))
                break
        for j in range(len(MIN[1])-1):
            if MIN[1][-1] <= y[i] and y[i] <= MAX[1][-1]:
                k = (MIN[0][-1]-MAX[0][-1])/(MIN[1][-1]-MAX[1][-1])
                xd.append(k*(y[i]-y[-1]) + MAX[0][-1])
                break
            elif MIN[1][j] <= y[i] and y[i] <= MIN[1][j+1]:
                xd.append(get_x(MIN,y[i],j))
                break
    return y.tolist(),xu,xd
# 上下限曲线末尾的对应半径处


def NEW_mr(m1,r1,m2,r2):
    '''上下限曲线到最大质量部分'''
    y1, xu1, xd1 = new_mr(m1, r1, m2, r2)
    # 无超出最大曲线部分直接返回
    if m1[-1] == max(m1) and m2[-1] == max(m2):
        return y1, xu1, xd1
    '''超出最大质量部分'''
    min_m = min(m1[-1],m2[-1])
    y2 = np.linspace(min_m,max(m1,m2),1000)
    # 最大质量对应半径
    r1_max, r2_max = r1[np.argwhere(m1==max(m1))[0]], r2[np.argwhere(m2==max(m2))[0]] 
    k1 = (r1_max-r2_max)/(max(r1)-max(r2)) # 最大质量对应的上边界斜率
    k2 = (r1[-1]-r2[-1])/(m1[-1]-m2[-1])   # 超出最大质量部分对应的下边界斜率
    for i in range(len(y2)):
        '''曲线是否有超出最大质量部分'''
    return

# 求N阶导数
class N_Derivative():
    
    def __init__(self, optimize=True):
        self.is_fit = False
        self.X, self.Y = None, None
        self.n = None
        self.optimize = optimize
    # 输入求导数据
    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.is_fit = True
    # 求n阶导
    def get_nD(self, n):
        if not self.is_fit:
            print('Not data.')
            return
        elif len(self.X) < n+1:
            print('Beyond the maximum derivative order.')
            return
        elif n == 0:
            return self.X, self.Y
        else:
            X, Y = self.X, self.Y
            while n > 0:
                Y = self.dydx(X, Y)
                if n%2 == 0:
                    X = X[:-1]
                else:
                    X = X[1:]
                n -= 1
            return X, Y
    # 基础求导
    def dydx(self, X, Y):
        d = []
        for i in range(len(X)-1):
            dy = Y[i+1]-Y[i]
            dx = X[i+1]-X[i]
            d.append(dy/dx)
        return np.array(d)
    # 比值 (n阶，N个间隔)
    def get_proportion(self, n, N):
        data_X, data_Y = self.get_nD(n)
        pro = []
        for i in range(len(data_Y)-N):
            pro.append((data_Y[i+N]-data_Y[i])/data_Y[i])
        return data_X[:-N], np.array(pro)




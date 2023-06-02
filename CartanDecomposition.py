import numpy as np
import copy
from scipy.linalg import sqrtm, logm, expm

from utils import PauliString, from_matrix, to_matrix

# Класс для KAK разложения однокубитного оператора
class OQF(object):
    def __init__(self, U):
        # polar in
        M_sq = self.Theta(U.T.conj()) @ U
        
        # spectral in
        D_sq, P = np.linalg.eig(M_sq)
        D = np.diag(np.sqrt(D_sq))
        # spectral out
        
        K = U @ P @ D.conj() @ P.T.conj()
        # polar out

        self.K1 = K @ P
        self.A = D
        self.K2 = P.T.conj()

    def Theta(self, U):
        Y = np.array([[0, -1j],
                      [1j, 0]], dtype = complex)
    
        return Y@U@Y
    
    def KAK(self):
        return self.K1, self.A, self.K2

# Класс для KAK разложения двухкубитного оператора
class TQF(object):
    def __init__(self, U):
        U_tilde = self.compute(U)
        
        # polar in
        M_sq = self.Theta(U_tilde.T.conj()) @ U_tilde
        
        # spectral in
        D_sq, P = np.linalg.eig(M_sq)
        if np.linalg.det(P)<0:
            P[:, [1, 2]] = P[:, [2, 1]]
            D_sq[[1, 2]] = D_sq[[2, 1]]
        
        D = np.sqrt(D_sq)
        if np.rint(np.sum(np.angle(D))/np.pi)>0:
            if np.angle(D[0])>0:
                D[0] = -D[0]
            else:
                D[1] = -D[1]
        elif np.rint(np.sum(np.angle(D))/np.pi)<0:
            if np.angle(D[0])<0:
                D[0] = -D[0]
            else:
                D[1] = -D[1]
        D_inv = np.diag(1/D)
        D = np.diag(D)
        # spectral out
        
        K = U_tilde @ P @ D_inv @ P.T.conj()
        # polar out
        
        self.K1 = self.uncompute(K @ P)
        self.K2 = self.uncompute(P.T.conj())
        self.A = self.uncompute(D)
    
    # Переход в "магический" базис
    def compute(self, U):
        B = np.array([[1, 1j, 0, 0],
                      [0, 0, 1j, 1],
                      [0, 0, 1j, -1],
                      [1, -1j, 0, 0]], dtype = complex)/np.sqrt(2)
        
        return B.T.conj() @ U @ B
    
    def Theta(self, U):
        return U.conj()
    
    # Переход из "магического" базиса
    def uncompute(self, U):
        B = np.array([[1, 1j, 0, 0],
                      [0, 0, 1j, 1],
                      [0, 0, 1j, -1],
                      [1, -1j, 0, 0]], dtype = complex)/np.sqrt(2)
    
        return B @ U @ B.T.conj()    
    
    def KAK(self):
        return self.K1, self.A, self.K2

# Класс "ядра" A3 разложения. Включает в себя Картановскую конволюцию разложения и унитарный переход в другой базис
# (U -(переход в другой базис)-> R^UR -(картановское разложение)-> R^K1R A R^K2R -(переход в исходный базис)-> K1 A K2)
# mask описывает инволюцию, R описывает переход 
class Core(object):
    def __init__(self, mask, R = None):
        self.mask = mask
        self.R = R
        
    
    def Theta(self, U):
        return self.mask @ U @ self.mask
    
    def compute(self, U):
        if self.R is None:
            return U
        else:
            return self.R.T.conj() @ U @ self.R
        
    def uncompute(self, U):
        if self.R is None:
            return U
        else:
            return self.R @ U @ self.R.T.conj()

# Класс A3 разложения. Принимает данное ядро
class CartDecompA3(object):
    def __init__(self, core):
        self.core = core
        
    def __call__(self, U):
        d = U.shape[0]
        
        M_sq = self.core.Theta(U.T.conj()) @ U
        
        D1_sq, P1 = spectral(self.core.compute(M_sq))

        D = self.core.uncompute(np.diag(np.sqrt(D1_sq)))
        P = self.core.uncompute(P1)
        
        K = U @ P @ D.conj() @ P.T.conj()
        
        K1 = K @ P
        A = D
        K2 = P.T.conj()
        
        K1 /= np.linalg.det(K1)**(1/d)
        A /= np.linalg.det(A)**(1/d)
        K2 /= np.linalg.det(K2)**(1/d)
        
        return K1, A, K2

# Взятие частичного следа по первому или последнему кубиту
def partial_trace(U, i = -1):
    H = logm(U)/1j
    
    d = U.shape[0]
    if i==-1:
        H_pt = np.trace(H.reshape([d//2, 2, d//2, 2]),
                        axis1 = 1, axis2 = 3)/2
    elif i==0:
        H_pt = np.trace(H.reshape([2, d//2, 2, d//2]), 
                        axis1 = 0, axis2 = 2)/2
        
    H_pt -= np.trace(H_pt)/(d//2) * np.eye(d//2)
    
#     if d==8:
#         c = from_matrix(H_pt)
#         print(c.terms[()] if () in c.terms else 'ok')
    
    return expm(1j*H_pt)

# Функция, выполняющая разбиение некоторого подпространства на два подпространства, 
# соответствующих с.з. +1,-1 матрицы Zn. 
# Принимает ортонормированный набор векторов и Zn, возвращает два набора ортонормированных векторов,
# соответствующих с.з. +1,-1 матрицы Zn
def separate(vec_list, Zn):    
    set_plus = list(filter(lambda x: not np.allclose(np.linalg.norm(x), 0), 
                           [(vec+Zn@vec)/2 for vec in vec_list]))
    
    if len(set_plus)>1:
        q, r = np.linalg.qr(np.array(set_plus).T)
        set_plus = [q.T[i] for i in range(len(r)) if not np.allclose(np.linalg.norm(r[i]), 0)]
    else:
        set_plus = [vec/np.linalg.norm(vec) for vec in set_plus]
        
    set_minus = list(filter(lambda x: not np.allclose(np.linalg.norm(x), 0), 
                            [(vec-Zn@vec)/2 for vec in vec_list]))
    
    if len(set_minus)>1:
        q, r = np.linalg.qr(np.array(set_minus).T)
        set_minus = [q.T[i] for i in range(len(r)) if not np.allclose(np.linalg.norm(r[i]), 0)]
    else:
        set_minus = [vec/np.linalg.norm(vec) for vec in set_minus]
    
    return set_plus, set_minus

# Функция спектрального разложения с учетом тензорной факторизации
def spectral(U):
    d = U.shape[0]
    n_qubits = int(np.log(d)/np.log(2))
    D, P = np.linalg.eig(U)
    
#     eigs1 = dict(zip(D, P.T))
    
    t = sorted(list(enumerate(np.angle(D))), key = lambda x: x[1], reverse = True)
    order, log_vals = zip(*t)
    
    pi_subspace = []
    zero_subspace = []
    pos_subspaces = {}
    
    cur_val = np.pi
    for i in range(d):
        if np.allclose(np.abs(log_vals[i]), np.pi):
            pi_subspace.append(order[i])
        elif np.allclose(log_vals[i], 0):
            zero_subspace.append(order[i])
        elif log_vals[i]>0:
            if np.allclose(log_vals[i], cur_val):
                pos_subspaces[cur_val].append(order[i])
            else:
                cur_val = log_vals[i]
                pos_subspaces[cur_val] = [order[i]]
    
    Zn = PauliString(((n_qubits-1, 'Z'),)).matrix()
    spectrum = {}
    if len(pi_subspace)>0:
        spectrum[np.pi], _ = separate(P.T[pi_subspace], Zn)
        
    if len(zero_subspace)>0:
        spectrum[0.], _ = separate(P.T[zero_subspace], Zn)
        
    Xn = PauliString(((n_qubits-1, 'X'),)).matrix()        
    for val in pos_subspaces:
        if len(pos_subspaces[val])>1:
            set_plus, set_minus = separate(P.T[pos_subspaces[val]], Zn)
            spectrum[val] = [vec for vec in set_plus]
            spectrum[-val] = [Xn@vec for vec in set_minus]
        else:
            i = pos_subspaces[val][0]
            coef = P.T[i].conj() @ Zn @ P.T[i]
            if coef>0:
                spectrum[val] = [P.T[i]]
            else:
                spectrum[-val] = [Xn@P.T[i]]
            
    val_list, vec_list = [], []
    for val in spectrum:
        for vec in spectrum[val]:
            val_list.append(np.exp(1j*val))
            vec_list.append(vec)
            val_list.append(np.exp(-1j*val))
            vec_list.append(Xn @ vec)
            
#     eigs2 = dict(zip(val_list, vec_list))
            
    return np.array(val_list), np.array(vec_list).T

# Квантовое разложение Шэннона
class QSD(object):
    def __init__(self, U):
        d = U.shape[0]
#         self.U = U/np.linalg.det(U)**(1/d)
        self.U = copy.copy(U)
        
        self.n_qubits = int(np.log(d)/np.log(2))
        if self.n_qubits>2:
            mask = PauliString(((self.n_qubits-1, 'Z'),)).matrix()
            R = (PauliString(((self.n_qubits-1, 'X'),)).matrix()\
                 +PauliString(((self.n_qubits-1, 'Z'),)).matrix())/np.sqrt(2)
            core1 = Core(mask, R)

            K1, self.A, K2 = CartDecompA3(core1)(self.U)
            
#             self.K1_ = copy.copy(K1)
#             self.K2_ = copy.copy(K2)

            mask = PauliString(((self.n_qubits-1, 'X'),)).matrix()
            core2 = Core(mask)

            T1, self.B1, T2 = CartDecompA3(core2)(K1)
            T3, self.B2, T4 = CartDecompA3(core2)(K2)
            
#             self.T1_ = copy.copy(T1)
#             self.T2_ = copy.copy(T2)
#             self.T3_ = copy.copy(T3)
#             self.T4_ = copy.copy(T4)

            self.K1 = QSD(partial_trace(T1))
            self.K2 = QSD(partial_trace(T2))
            self.K3 = QSD(partial_trace(T3))
            self.K4 = QSD(partial_trace(T4))
            
        else:
            self.K1, self.A, self.K2 = TQF(self.U).KAK()
    
    def components(self):
        if self.n_qubits>2:
            return self.K1.components()+[self.B1]\
                +self.K2.components()+[self.A]+self.K3.components()\
                +[self.B2]+self.K4.components()
        
        else:
            return [self.K1, self.A, self.K2]
import numpy as np
import torch
from torch import nn
from scipy.linalg import sqrtm, logm, expm
from openfermion.ops import QubitOperator
from openfermion.utils import count_qubits

from utils import PauliString, to_matrix, from_matrix

# Первый шаг алгоритма. Получение алгебры, содержащей строки Паули, входящие в Гамильтониан
def get_hamiltonian_algebra(hamiltonian):
    n_qubits = count_qubits(hamiltonian)
    hamiltonian_terms = [PauliString(term, n_qubits) for term in hamiltonian.terms]
    
    algebra_terms = []
    old_batch = [term for term in hamiltonian_terms]
    new_batch = []
    
    step = 0
    while (len(old_batch)>0) & (step<32):
        for term1 in old_batch:
            for term2 in hamiltonian_terms:
                if (not term1^term2) & ((term1*term2) not in algebra_terms) & ((term1*term2) not in new_batch):
                    new_batch.append(term1*term2)

        algebra_terms += old_batch
        old_batch = [term for term in new_batch]
        new_batch = []
        step += 1
        
    return algebra_terms

# Второй шаг алгоритма. Картановское разложение алгебры
def get_cartan_decomposition(algebra):
    def theta(U):
        return -U.T

    l_list = []
    m_list = []

    for term in algebra:
        P = term.matrix()
        if np.allclose(theta(P), P):
            l_list.append(term)
        elif np.allclose(theta(P), -P):
            m_list.append(term)
            
    return m_list, l_list

# Второй шаг алгоритма. Получение одной из возможных картановских алгебр (нахождение максимальной абелевой подалгебры)
def get_cartan_algebra(m_list):
    h1_list = []
    h_list = []
    max_size = 0

    for i, term in enumerate(m_list):
        h1_list.append(term)
        size = 0

        for j in range(i+1, len(m_list)):
            if all([(m_list[j]^term)==1 for term in h1_list]):
                h1_list.append(m_list[j])
                size += 1
        if max_size<size:
            h_list = [term for term in h1_list]
            
    return h_list

# Третий шаг алгоритма. Получение коэффициентов a и gamma
def decompose(hamiltonian, l_list, h_list):
    device = 'cpu'
    K_list = [torch.tensor(term.matrix(), dtype=torch.cfloat, requires_grad = False) for term in l_list]
    def K(a):
        I = torch.eye(K_list[0].shape[0], dtype=torch.cfloat, requires_grad = False)
        for i, k in enumerate(K_list):
            if i>0:
                K = (torch.cos(a[i])*I-1j*torch.sin(a[i])*k) @ K
            else:
                K = (torch.cos(a[i])*I-1j*torch.sin(a[i])*k)
        return K

    gamma = torch.tensor(2*np.pi*torch.rand(len(h_list), device=device), requires_grad = False)
    for i, term in enumerate(h_list):
        if i>0:
            v += gamma[i]*torch.tensor(term.matrix(), dtype=torch.cfloat, requires_grad = False)
        else:
            v = gamma[i]*torch.tensor(term.matrix(), dtype=torch.cfloat, requires_grad = False)
    
    H = torch.tensor(to_matrix(hamiltonian), dtype=torch.cfloat, requires_grad = False)
    
    
    a = nn.parameter.Parameter(2*np.pi*torch.rand(len(K_list), device=device), requires_grad = True)
    optimizer = torch.optim.LBFGS([a], 
                                  max_iter=100, 
                                  history_size=10, 
                                  tolerance_grad=1e-09,
                                  tolerance_change=1e-12,
                                  line_search_fn = 'strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = torch.trace(H @ K(a) @ v @ K(a).T.conj())
        loss.backward()
        return loss
    optimizer.step(closure)
    
    h = from_matrix((K(a).T.conj() @ H @ K(a)).detach().numpy())
    h.compress(abs_tol = 1e-6)
    gamma = np.array([h.terms[term.term] for term in h_list])
    
    return a.detach().numpy(), gamma

# Класс эмуляции эволюции гамильтониана через Картановское разложение
class CartanEmulation(object):
    def __init__(self, l_list, h_list, a, gamma):
        self.l_terms = l_list
        self.h_terms = h_list
        self.l_coefs = a
        self.h_coefs = gamma
        
        self.n_qubits = l_list[0].n_qubits
        
    def __call__(self, t):
        id_matrix = np.eye(2**self.n_qubits)
        for i, term in enumerate(self.l_terms):
            if i>0:
                K = (np.cos(self.l_coefs[i])*id_matrix-1j*np.sin(self.l_coefs[i])*term.matrix()) @ K
            else:
                K = np.cos(self.l_coefs[i])*id_matrix-1j*np.sin(self.l_coefs[i])*term.matrix()
            
        for i, term in enumerate(self.h_terms):
            if i>0:
                A = (np.cos(self.h_coefs[i]*t)*id_matrix-1j*np.sin(self.h_coefs[i]*t)*term.matrix()) @ A
            else:
                A = np.cos(self.h_coefs[i]*t)*id_matrix-1j*np.sin(self.h_coefs[i]*t)*term.matrix()
            
        return K@A@K.T.conj()
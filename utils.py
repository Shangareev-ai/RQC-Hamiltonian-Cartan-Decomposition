import numpy as np
from openfermion.ops import QubitOperator
from openfermion.utils import count_qubits

_PAULI_OPERATOR_PRODUCTS = {('I', 'I'): ('I'),
                            ('I', 'X'): ('X'),
                            ('X', 'I'): ('X'),
                            ('I', 'Y'): ('Y'),
                            ('Y', 'I'): ('Y'),
                            ('I', 'Z'): ('Z'),
                            ('Z', 'I'): ('Z'),
                            ('X', 'X'): ('I'),
                            ('Y', 'Y'): ('I'),
                            ('Z', 'Z'): ('I'),
                            ('X', 'Y'): ('Z'),
                            ('X', 'Z'): ('Y'),
                            ('Y', 'X'): ('Z'),
                            ('Y', 'Z'): ('X'),
                            ('Z', 'X'): ('Y'),
                            ('Z', 'Y'): ('X')}

def _simplify(term):
    if not term:
        return coefficient, term

    term = sorted(term, key=lambda factor: factor[0])

    new_term = []
    left_factor = term[0]
    for right_factor in term[1:]:
        left_index, left_action = left_factor
        right_index, right_action = right_factor

        if left_index == right_index:
            new_action = _PAULI_OPERATOR_PRODUCTS[left_action, right_action]
            left_factor = (left_index, new_action)

        else:
            if left_action != 'I':
                new_term.append(left_factor)
            left_factor = right_factor

    if left_factor[1] != 'I':
        new_term.append(left_factor)

    return tuple(new_term)

class PauliString(object):
    """PauliString class provides representation of Pauli string operators (e.g. 'XX', 'IY')
       The signs *^ overloaded with multiplication and commutation relationship
    """
    def __init__(self, term, n_qubits = None):
        """Initializtion of Pauli string
           
           Args:
               term (tuple): Tuple of actions on qubit (e.g. 'IXIYZ' is ((1, 'X'), (3, 'Y'), (4, 'Z')))
               n_qubits (int): Number of qubits, on which operator is acting
        """
        self.term = tuple([act for act in term])
        
        if n_qubits is not None:
            self.n_qubits = n_qubits
        else:
            self.n_qubits = max([i for i, act in term])+1 if len(term)>0 else 0
        
    def __str__(self):
        """String representation of operator
        """
        last_index = -1
        pauliCode = []

        for i, p in self.term:
            if i>last_index+1:
                pauliCode.extend(['I' for j in range(last_index+1, i)])

            pauliCode.append(p)

            last_index = i

        if last_index+1<self.n_qubits:
            pauliCode.extend(['I' for j in range(last_index+1, self.n_qubits)])


        return ''.join(pauliCode)
    
    def matrix(self):
        """Numpy array representation of operator 
        """
        mat = 1.
        for act in self.__str__():
            if act=='I':
                mat = np.kron(mat, np.array([[1, 0],
                                             [0, 1]], dtype = complex))
            elif act=='X':
                mat = np.kron(mat, np.array([[0, 1],
                                             [1, 0]], dtype = complex))
            elif act=='Y':
                mat = np.kron(mat, np.array([[0, -1j],
                                             [1j, 0]], dtype = complex))
            else:
                mat = np.kron(mat, np.array([[1, 0],
                                             [0, -1]], dtype = complex))
                
        return mat
    
    def __mul__(self, pauli_st):
        """Multiplication operator
        
           Args:
               pauli_st (PauliString)
               
           Returns (PauliString): Dot product of operators (without sign)
        """
        res_str = [_PAULI_OPERATOR_PRODUCTS[prod] for prod in zip(self.__str__(), str(pauli_st))]
        
        return PauliString.from_str(res_str)
                
    def __xor__(self, pauli_st):
        """Whether operators commutes
        
           Args:
               pauli_st (PauliString)
               
           Returns (bool): True if operators commutes, False otherwise
        """
        t = 0
        for act1, act2 in zip(self.__str__(), str(pauli_st)):
            if (act1!='I') and (act2!='I') and (act1!=act2):
                t += 1
        
        return not bool(t%2)
    
    def __eq__(self, pauli_st):
        """Whether operators is equal
        """
        return (self.n_qubits==pauli_st.n_qubits) and (self.term == pauli_st.term)
    
    def from_str(pauli_str):
        """Get PauliSting operator from string
        
           Args:
               String representation of operator (e.g. 'IZIYX')
               
           Returns (PauliString): PauliString representation of a string
        """
        term = []
        n_qubits = len(pauli_str)
        for i, act in enumerate(pauli_str):
            if act!='I':
                term.append((i, act))
        
        return PauliString(tuple(term), n_qubits)

def to_matrix(qubit_op):
    """Returns numpy array representation of openfermion QubitOperator
    
       Args:
           qubit_op (openfermion.ops.QubitOperator)
           
       Returns (numpy.array): Matrix of operator
    """
    
    n_qubits = count_qubits(qubit_op)
    matrix = np.zeros((2**n_qubits, 2**n_qubits), dtype = complex)
    
    for term, coef in qubit_op.terms.items():
        matrix += coef*PauliString(term, n_qubits).matrix()
        
    return matrix

def from_matrix(matrix):
    """Return openfermion QubitOperator of hermitian matrix
    
       Args:
           matrix (numpy.array)
           
       Returns (openfermion.ops.QubitOperator): QubitOperator represented by given matrix
    """
    qubit_op = QubitOperator()
    n_qubits = int(np.log(matrix.shape[0])/np.log(2))
    
    pauli_dict = {'01':'X', '10':'Y', '11':'Z'}
    for i in range(4**n_qubits):
        b = format(i, '0'+str(2*n_qubits)+'b')
        term = tuple([(j, pauli_dict[b[2*j:2*j+2]]) for j in range(n_qubits) if b[2*j:2*j+2]!='00'])
        
        coef = np.trace(matrix @ PauliString(term, n_qubits).matrix())/2**n_qubits
        
        qubit_op += QubitOperator(term, coef)
        
    return qubit_op

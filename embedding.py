import pennylane as qml
from pennylane import numpy as np

N_layers = 2

# exp(ixZ) gate
def exp_Z(x, wires, inverse=False):
  if inverse == False:
    qml.RZ(-2 * x, wires=wires)
  elif inverse == True:
    qml.RZ(2 * x, wires=wires)

# exp(ixZZ) gate
def exp_ZZ1(x, wires, inverse=False):
  if inverse == False:
    qml.CNOT(wires=wires)
    qml.RZ(-2 * x, wires=wires[1])
    qml.CNOT(wires=wires)
  elif inverse == True:
    qml.CNOT(wires=wires)
    qml.RZ(2 * x, wires=wires[1])
    qml.CNOT(wires=wires)

# exp(i(pi - x1)(pi - x2)ZZ) gate
def exp_ZZ2(x1, x2, wires, inverse=False):
  if inverse == False:
    qml.CNOT(wires=wires)
    qml.RZ(-2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
    qml.CNOT(wires=wires)
  elif inverse == True:
    qml.CNOT(wires=wires)
    qml.RZ(2 * (np.pi - x1) * (np.pi - x2), wires=wires[1])
    qml.CNOT(wires=wires)


# Quantum Embedding 1 for model 1 (Conventional ZZ feature embedding)
def QuantumEmbedding_2qubits(input):
    for i in range(N_layers):
        qml.Hadamard(wires=0)
        exp_Z(input[0], wires=0)
        qml.Hadamard(wires=1)
        exp_Z(input[1], wires=1)
        exp_ZZ2(input[0], input[1], wires=[0, 1])

def QuantumEmbedding_2qubits_inverse(input):
    for i in range(N_layers):
        exp_ZZ2(input[0], input[1], wires=[0, 1])
        qml.Barrier()
        exp_Z(input[0], wires=0, inverse=True)
        qml.Hadamard(wires=0)
        exp_Z(input[1], wires=1, inverse=True)
        qml.Hadamard(wires=1)

def QuantumEmbedding_4qubits(input):
    for i in range(N_layers):
        qml.Hadamard(wires=0)
        exp_Z(input[0], wires=0)
        qml.Hadamard(wires=1)
        exp_Z(input[1], wires=1)
        qml.Hadamard(wires=2)
        exp_Z(input[2], wires=2)
        qml.Hadamard(wires=3)
        exp_Z(input[3], wires=3)
        ##-------------------##
        exp_ZZ2(input[0], input[1], wires=[0, 1])
        exp_ZZ2(input[1], input[2], wires=[1, 2])
        exp_ZZ2(input[2], input[3], wires=[2, 3])

def QuantumEmbedding_4qubits_inverse(input):
    for i in range(N_layers):
        exp_ZZ2(input[2], input[3], wires=[2, 3])
        exp_ZZ2(input[1], input[2], wires=[1, 2])
        exp_ZZ2(input[0], input[1], wires=[0, 1])
        ##-------------------##
        qml.Barrier()
        exp_Z(input[3], wires=3, inverse=True)
        qml.Hadamard(wires=3)
        exp_Z(input[2], wires=2, inverse=True)
        qml.Hadamard(wires=2)
        exp_Z(input[1], wires=1, inverse=True)
        qml.Hadamard(wires=1)
        exp_Z(input[0], wires=0, inverse=True)
        qml.Hadamard(wires=0)


def QuantumEmbedding_8qubits(input):
    for i in range(N_layers):
        qml.Hadamard(wires=0)
        exp_Z(input[:,0], wires=0)
        qml.Hadamard(wires=1)
        exp_Z(input[:,1], wires=1)
        qml.Hadamard(wires=2)
        exp_Z(input[:,2], wires=2)
        qml.Hadamard(wires=3)
        exp_Z(input[:,3], wires=3)
        qml.Hadamard(wires=4)
        exp_Z(input[:,4], wires=4)
        qml.Hadamard(wires=5)
        exp_Z(input[:,5], wires=5)
        qml.Hadamard(wires=6)
        exp_Z(input[:,6], wires=6)
        qml.Hadamard(wires=7)
        exp_Z(input[:,7], wires=7)
        ##-------------------##
        exp_ZZ2(input[:,0], input[:,1], wires=[0, 1])
        exp_ZZ2(input[:,1], input[:,2], wires=[1, 2])
        exp_ZZ2(input[:,2], input[:,3], wires=[2, 3])
        exp_ZZ2(input[:,3], input[:,4], wires=[3, 4])
        exp_ZZ2(input[:,4], input[:,5], wires=[4, 5])
        exp_ZZ2(input[:,5], input[:,6], wires=[5, 6])
        exp_ZZ2(input[:,6], input[:,7], wires=[6, 7])

def QuantumEmbedding_8qubits_inverse(input):
    for i in range(N_layers):
        exp_ZZ2(input[6], input[7], wires=[6, 7])
        exp_ZZ2(input[5], input[6], wires=[5, 6])
        exp_ZZ2(input[4], input[5], wires=[4, 5])
        exp_ZZ2(input[3], input[4], wires=[3, 4])
        exp_ZZ2(input[2], input[3], wires=[2, 3])
        exp_ZZ2(input[1], input[2], wires=[1, 2])
        exp_ZZ2(input[0], input[1], wires=[0, 1])
        ##-------------------##
        qml.Barrier()
        exp_Z(input[7], wires=7, inverse=True)
        qml.Hadamard(wires=7)
        exp_Z(input[6], wires=6, inverse=True)
        qml.Hadamard(wires=6)
        exp_Z(input[5], wires=5, inverse=True)
        qml.Hadamard(wires=5)
        exp_Z(input[4], wires=4, inverse=True)
        qml.Hadamard(wires=4)
        exp_Z(input[3], wires=3, inverse=True)
        qml.Hadamard(wires=3)
        exp_Z(input[2], wires=2, inverse=True)
        qml.Hadamard(wires=2)
        exp_Z(input[1], wires=1, inverse=True)
        qml.Hadamard(wires=1)
        exp_Z(input[0], wires=0, inverse=True)
        qml.Hadamard(wires=0)


def QuantumEmbedding_16qubits(input):
    for i in range(N_layers):
        qml.Hadamard(wires=0)
        exp_Z(input[:,0], wires=0)
        qml.Hadamard(wires=1)
        exp_Z(input[:,1], wires=1)
        qml.Hadamard(wires=2)
        exp_Z(input[:,2], wires=2)
        qml.Hadamard(wires=3)
        exp_Z(input[:,3], wires=3)
        qml.Hadamard(wires=4)
        exp_Z(input[:,4], wires=4)
        qml.Hadamard(wires=5)
        exp_Z(input[:,5], wires=5)
        qml.Hadamard(wires=6)
        exp_Z(input[:,6], wires=6)
        qml.Hadamard(wires=7)
        exp_Z(input[:,7], wires=7)
        qml.Hadamard(wires=8)
        exp_Z(input[:,8], wires=8)
        qml.Hadamard(wires=9)
        exp_Z(input[:,9], wires=9)
        qml.Hadamard(wires=10)
        exp_Z(input[:,10], wires=10)
        qml.Hadamard(wires=11)
        exp_Z(input[:,11], wires=11)
        qml.Hadamard(wires=12)
        exp_Z(input[:,12], wires=12)
        qml.Hadamard(wires=13)
        exp_Z(input[:,13], wires=13)
        qml.Hadamard(wires=14)
        exp_Z(input[:,14], wires=14)
        qml.Hadamard(wires=15)
        exp_Z(input[:,15], wires=15)
        ##-------------------##
        exp_ZZ2(input[:,0], input[:,1], wires=[0, 1])
        exp_ZZ2(input[:,1], input[:,2], wires=[1, 2])
        exp_ZZ2(input[:,2], input[:,3], wires=[2, 3])
        exp_ZZ2(input[:,3], input[:,4], wires=[3, 4])
        exp_ZZ2(input[:,4], input[:,5], wires=[4, 5])
        exp_ZZ2(input[:,5], input[:,6], wires=[5, 6])
        exp_ZZ2(input[:,6], input[:,7], wires=[6, 7])
        exp_ZZ2(input[:,7], input[:,8], wires=[7, 8])
        exp_ZZ2(input[:,8], input[:,9], wires=[8, 9])
        exp_ZZ2(input[:,9], input[:,10], wires=[9, 10])
        exp_ZZ2(input[:,10], input[:,11], wires=[10, 11])
        exp_ZZ2(input[:,11], input[:,12], wires=[11, 12])
        exp_ZZ2(input[:,12], input[:,13], wires=[12, 13])
        exp_ZZ2(input[:,13], input[:,14], wires=[13, 14])
        exp_ZZ2(input[:,14], input[:,15], wires=[14, 15])
        
def QuantumEmbedding_16qubits_inverse(input):
    for i in range(N_layers):
        exp_ZZ2(input[14], input[15], wires=[14, 15])
        exp_ZZ2(input[13], input[14], wires=[13, 14])
        exp_ZZ2(input[12], input[13], wires=[12, 13])
        exp_ZZ2(input[11], input[12], wires=[11, 12])
        exp_ZZ2(input[10], input[11], wires=[10, 11])
        exp_ZZ2(input[9], input[10], wires=[9, 10])
        exp_ZZ2(input[8], input[9], wires=[8, 9])
        exp_ZZ2(input[7], input[8], wires=[7, 8])
        exp_ZZ2(input[6], input[7], wires=[6, 7])
        exp_ZZ2(input[5], input[6], wires=[5, 6])
        exp_ZZ2(input[4], input[5], wires=[4, 5])
        exp_ZZ2(input[3], input[4], wires=[3, 4])
        exp_ZZ2(input[2], input[3], wires=[2, 3])
        exp_ZZ2(input[1], input[2], wires=[1, 2])
        exp_ZZ2(input[0], input[1], wires=[0, 1])
        ##-------------------##
        qml.Barrier()
        exp_Z(input[15], wires=15, inverse=True)
        qml.Hadamard(wires=15)
        exp_Z(input[14], wires=14, inverse=True)
        qml.Hadamard(wires=14)
        exp_Z(input[13], wires=13, inverse=True)
        qml.Hadamard(wires=13)
        exp_Z(input[12], wires=12, inverse=True)
        qml.Hadamard(wires=12)
        exp_Z(input[11], wires=11, inverse=True)
        qml.Hadamard(wires=11)
        exp_Z(input[10], wires=10, inverse=True)
        qml.Hadamard(wires=10)
        exp_Z(input[9], wires=9, inverse=True)
        qml.Hadamard(wires=9)
        exp_Z(input[8], wires=8, inverse=True)
        qml.Hadamard(wires=8)
        exp_Z(input[7], wires=7, inverse=True)
        qml.Hadamard(wires=7)
        exp_Z(input[6], wires=6, inverse=True)
        qml.Hadamard(wires=6)
        exp_Z(input[5], wires=5, inverse=True)
        qml.Hadamard(wires=5)
        exp_Z(input[4], wires=4, inverse=True)
        qml.Hadamard(wires=4)
        exp_Z(input[3], wires=3, inverse=True)
        qml.Hadamard(wires=3)
        exp_Z(input[2], wires=2, inverse=True)
        qml.Hadamard(wires=2)
        exp_Z(input[1], wires=1, inverse=True)
        qml.Hadamard(wires=1)
        exp_Z(input[0], wires=0, inverse=True)
        qml.Hadamard(wires=0)

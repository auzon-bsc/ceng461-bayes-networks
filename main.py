from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

import numpy as np

# 1. Variable elimination (using the pgmpy package)
bayesNet = BayesianNetwork()

bayesNet.add_node("A")
bayesNet.add_node("B")
bayesNet.add_node("C")
bayesNet.add_node("D")
bayesNet.add_node("E")

bayesNet.add_edge("A", "B")
bayesNet.add_edge("A", "C")
bayesNet.add_edge("B", "D")
bayesNet.add_edge("C", "D")
bayesNet.add_edge("C", "E")

# create conditional probability distributions
# when giving values, FALSE ones should be given first
cpd_A = TabularCPD('A', 2, values=[[.8], [.2]])
cpd_B = TabularCPD('B', 2, values=[[.8, .2], [.2, .8]],
                   evidence=['A'], evidence_card=[2])
cpd_C = TabularCPD('C', 2, values=[[.95, .8], [.05, .2]],
                   evidence=['A'], evidence_card=[2])
cpd_D = TabularCPD('D', 2, values=[[.95, .2, .2, .2], [.05, .8, .8, .8]],
                   evidence=['B', 'C'], evidence_card=[2, 2])
cpd_E = TabularCPD('E', 2, values=[[.4, .2], [.6, .8]],
                   evidence=['C'], evidence_card=[2])

bayesNet.add_cpds(cpd_A, cpd_B, cpd_C, cpd_D, cpd_E)

if bayesNet.check_model():
    print("Model is correct.")

solver = VariableElimination(bayesNet)
D = solver.query(variables=['D'])
A = solver.query(variables=['A'])
D_A0 = solver.query(variables=['D'], evidence={'A': 0})
E_B0 = solver.query(variables=['E'], evidence={'B': 0})
A_D1E0 = solver.query(variables=['A'], evidence={'D': 1, 'E': 0})
B_A1 = solver.query(variables=['B'], evidence={'A': 1})
E_A1 = solver.query(variables=['E'], evidence={'A': 1})

# [0] = P(-), [1] = P(+)

# P(+D)
D1 = (D.values)[1]
# $ P(+D,-A) = P(P(+D|-A) * P(-A))
D1A0 = (D_A0.values)[1] * (A.values)[0]
# $ P(+E|-B)
E1_B0 = (E_B0.values)[1]
# $ P(+A|+D,-E)
A1_D1E0 = (A_D1E0.values)[1]
# $ P(+B,-E|+A) = P(+B|+A) * P(-E|+A)
B1E0_A1 = (B_A1.values)[1] * (E_A1.values)[0]

print(f"")
print(f"1. Results using variable elimination:")
print(f"======================================")
print(f"P(+D) == {round(D1, 3)}")
print(f"P(+D,-A) == {round(D1A0, 3)}")
print(f"P(+E|-B) == {round(E1_B0, 3)}")
print(f"P(+A|+D,-E) == {round(A1_D1E0, 3)}")
print(f"P(+B,-E|+A) == {round(B1E0_A1, 3)}")

# 2. Monte Carlo technique

# Sampling
N = 1000000
A, B, C, D, E = [], [], [], [], []
for i in range(N):
    a = np.random.rand() < 0.2
    A.append(a)
    if a == True:
        b = np.random.rand() < 0.8
        c = np.random.rand() < 0.2
    else:
        b = np.random.rand() < 0.2
        c = np.random.rand() < 0.05
    B.append(b)
    C.append(c)
    if b == True:
        if c == True:
            d = np.random.rand() < 0.8
        else:
            d = np.random.rand() < 0.8
    else:
        if c == True:
            d = np.random.rand() < 0.8
        else:
            d = np.random.rand() < 0.05
    D.append(d)
    if c == True:
        e = np.random.rand() < 0.8
    else:
        e = np.random.rand() < 0.6
    E.append(e)

# Queries
DA = []
E_B0 = []
A_D1E0 = []
B_A1 = []
E_A1 = []
for i in range(N):
    DA.append(D[i] and not A[i])
    if not B[i]:
        E_B0.append(E[i])
    if D[i] and not E[i]:
        A_D1E0.append(A[i])
    if A[i]:
        B_A1.append(B[i])
        E_A1.append(E[i])

# Probability calculations

# $ P(+D)
D1 = sum(D)/len(D)
# $ P(+D|-A)
D1A0 = sum(DA)/len(DA)
# $ P(+E|-B)
E1_B0 = sum(E_B0)/len(E_B0)
# $ P(+A|+D,-E)
A1_D1E0 = sum(A_D1E0)/len(A_D1E0)
# $ P(+B,-E|+A) = P(+B|+A) * P(-E|+A)
B1E0_A1 = sum(B_A1)/len(B_A1) * (1 - sum(E_A1)/len(E_A1))

print(f"")
print(f"2. Results using monte carlo:")
print(f"======================================")
print(f"P(+D) == {round(D1, 3)}")
print(f"P(+D,-A) == {round(D1A0, 3)}")
print(f"P(+E|-B) == {round(E1_B0, 3)}")
print(f"P(+A|+D,-E) == {round(A1_D1E0, 3)}")
print(f"P(+B,-E|+A) == {round(B1E0_A1, 3)}")

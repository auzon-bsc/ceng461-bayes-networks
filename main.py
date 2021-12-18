from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

bayesNet = BayesianModel()
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

# when giving values, FALSE ones should be given first
cpd_A = TabularCPD('A', 2, values=[[.8], [.2]])
cpd_B = TabularCPD('S', 2, values=[[0.8, .2], [.2, .8]],
                   evidence=['A'], evidence_card=[1, 1])
cpd_C = TabularCPD('S', 2, values=[[0.95, .8], [.05, .2]],
                   evidence=['A'], evidence_card=[1, 1])


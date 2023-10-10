import qutip as qt

from chencrafts.bsqubits.QEC_trajectory.node import StateNode, StateEnsemble
from chencrafts.bsqubits.QEC_trajectory.edge import PropagatorEdge, MeasurementEdge, Edge

from typing import List    

class EvolutionGraph:
    def __init__(self):
        self.nodes: List[StateNode] = []
        self.edges: List[Edge] = []

    def add_node(self, node: StateNode):
        node.assign_index(self.node_num)
        self.nodes.append(node)

    def add_edge(self, edge: Edge, init_node: StateNode, final_node: StateNode):
        edge.assign_index(self.edge_num)

        init_node.add_out_edges(edge)
        edge.connect(
            init_node, final_node,
        )

        self.edges.append(edge)

    @property
    def node_num(self):
        return len(self.nodes)
    
    @property
    def edge_num(self):
        return len(self.edges)
    

class EvolutionTree(EvolutionGraph):

    """
    If it's a tree, then the final states are always the new nodes that 
    have not been traversed.
    """

    def _evolve_single_step(self, initial_ensemble: StateEnsemble) -> StateEnsemble:
        final_ensemble = StateEnsemble()

        for node in initial_ensemble:
            if node.out_edges == []:
                raise RuntimeError(
                "The node has no out edges. "
                "Possibly the evolution reaches the final state."
            )
            for edge in node.out_edges:
                edge.evolve()
                final_ensemble.add(edge.final_state)

        return final_ensemble

    def evolve(
        self,
        initial_ensemble: StateEnsemble,
        steps: int,
    ) -> StateEnsemble:
        """
        Evolve the initial ensemble for a number of steps

        Parameters
        ----------
        initial_ensemble : StateEnsemble
            The initial ensemble
        steps : int
            Number of steps to evolve

        Returns
        -------
        final_ensemble : StateEnsemble
            The final ensemble after evolution
        """
        
        current_ensemble = initial_ensemble

        for stp in range(steps):
            try:
                current_ensemble = self._evolve_single_step(current_ensemble)
            except RuntimeError:
                print(f"The evolution stops at step {stp}.")
                break

        return current_ensemble
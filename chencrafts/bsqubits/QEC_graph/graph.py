import qutip as qt
import networkx as nx

from chencrafts.bsqubits.QEC_graph.node import StateNode, StateEnsemble
from chencrafts.bsqubits.QEC_graph.edge import PropagatorEdge, MeasurementEdge, Edge

from typing import List    

class EvolutionGraph:
    def __init__(self):
        self.nodes: List[StateNode] = []
        self.edges: List[Edge] = []

    def add_node(self, node: StateNode):
        node.assign_index(self.node_num)
        self.nodes.append(node)

    def add_edge_connect(self, edge: Edge, init_node: StateNode, final_node: StateNode):
        """
        Add an edge and connect it to the initial node and the final node,
        which are already in the graph.
        """
        assert init_node in self.nodes
        assert final_node in self.nodes

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
    
    def to_nx(self) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from([node.to_nx() for node in self.nodes])
        nx_graph.add_edges_from([edge.to_nx() for edge in self.edges])

        return nx_graph
    
    def clear_evolution_data(
        self, 
        exclude: List[StateNode] = [],
    ):
        for node in self.nodes:
            if node not in exclude:
                node.clear_evolution_data()
    

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
                final_ensemble.append(edge.final_state)

        return final_ensemble
    
    def _move_single_step(self, initial_ensemble: StateEnsemble) -> StateEnsemble:
        """
        In the tree, the node will not be traversed twice when evolving.
        By moving on the tree, the evolution data can be retrieved.
        This function return the ensemble at the next step without evolving
        """
        next_ensemble = StateEnsemble()

        for node in initial_ensemble:
            for edge in node.out_edges:
                next_ensemble.append(edge.final_state)

        return next_ensemble

    def ensemble_at(self, step: int) -> StateEnsemble:
        """
        In the tree, the node will not be traversed twice when evolving.
        By moving on the tree, the evolution data can be retrieved.
        This function return the ensemble at the next step at the given step
        """
        current_ensemble = StateEnsemble([self.nodes[0]])

        for stp in range(step):
            try:
                current_ensemble = self._move_single_step(current_ensemble)
            except RuntimeError:
                print(f"The evolution stops at step {stp}.")
                break

        return current_ensemble
        

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
        self.clear_evolution_data(exclude=initial_ensemble)
        
        current_ensemble = initial_ensemble

        for stp in range(steps):
            try:
                current_ensemble = self._evolve_single_step(current_ensemble)
            except RuntimeError:
                print(f"The evolution stops at step {stp}.")
                break

        return current_ensemble
    
    def evolve_all(self):
        return self.evolve(StateEnsemble([self.nodes[0]]), 9999)
    
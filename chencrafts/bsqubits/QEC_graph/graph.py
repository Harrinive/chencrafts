import qutip as qt

from chencrafts.bsqubits.QEC_graph.node import StateEnsemble

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from chencrafts.bsqubits.QEC_graph.node import Node
    from chencrafts.bsqubits.QEC_graph.edge import Edge

class EvolutionGraph:
    def __init__(self):
        self.nodes: List["Node"] = []
        self.edges: List["Edge"] = []

    def add_node(self, node: "Node"):
        node.assign_index(self.node_num)
        self.nodes.append(node)

    def add_edge_connect(self, edge: "Edge", init_node: "Node", final_node: "Node"):
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
    
    def to_nx(self) -> "nx.DiGraph":
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is a optional dependency for QEC_graph module."
                "Please install it via 'pip install networkx' or 'conda install networkx'."
            )

        nx_graph = nx.DiGraph()
        nx_graph.add_nodes_from([node.to_nx() for node in self.nodes])
        nx_graph.add_edges_from([edge.to_nx() for edge in self.edges])

        return nx_graph
    
    def clear_evolution_data(
        self, 
        exclude: List["Node"] = [],
    ):
        for node in self.nodes:
            if node not in exclude:
                node.clear_evolution_data()
    

class EvolutionTree(EvolutionGraph):
    """
    If it's a tree, then the final states are always the new nodes that 
    have not been traversed.
    """

    def _traverse_single_step(
        self, 
        initial_ensemble: StateEnsemble,
        evolve: bool = True,
    ) -> StateEnsemble:
        final_ensemble = StateEnsemble()

        # check if the evolution reaches the final state, namely no node
        # has out edges
        if initial_ensemble.no_further_evolution:
            return final_ensemble

        # evolve
        for node in initial_ensemble:
            if node.terminated:
                # a manually terminated node will not be evolved and it
                # will always be in the final ensemble, unchanged
                final_ensemble.append(node)
                continue

            if node.out_edges == []:
                # this node has no out edges (and not terminated),
                # usually because the occupation 
                # of the state (normalization factor) is very small
                continue

            for edge in node.out_edges:
                if evolve:
                    edge.evolve()

                if edge.final_state not in final_ensemble:
                    # the final state has not been traversed
                    final_ensemble.append(edge.final_state)

        return final_ensemble
    
    def traverse(
        self,
        steps: int,
        initial_ensemble: StateEnsemble = None,
        evolve: bool = True,
    ):
        if initial_ensemble is None:
            current_ensemble = StateEnsemble([self.nodes[0]])
        else:
            current_ensemble = initial_ensemble

        for stp in range(steps):
            current_ensemble = self._traverse_single_step(
                current_ensemble, evolve=evolve
            )
            if current_ensemble.no_further_evolution:
                if stp+1 < steps:
                    print(f"The evolution stops earlier at step {stp+1}")
                break

        return current_ensemble
    
    def attr_by_step(
        self, 
        attr: str, 
        handle_error: bool = True,
        *args, 
        **kwargs,
    ) -> List:
        """
        Return the attribute of the final state at each step. If callable,
        then the attribute is the result of calling the method with 
        *args and **kwargs.
        """
        def _get_attr(ensemble: "StateEnsemble"):
            try:
                current_attr = getattr(ensemble, attr)
                if callable(current_attr):
                    return current_attr(*args, **kwargs)
                else:
                    return current_attr
                
            except Exception as e:
                if handle_error:
                    return None
                else:
                    raise e

        current_ensemble = StateEnsemble([self.nodes[0]])
        attr_by_stp = [_get_attr(current_ensemble)]

        while True:
            current_ensemble = self._traverse_single_step(
                current_ensemble, evolve=False
            )
            attr_by_stp.append(_get_attr(current_ensemble))
            if current_ensemble.no_further_evolution:
                break
            
        return attr_by_stp
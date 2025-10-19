import networkx as nx

from graph_utils import HierarchyCatalog
from graph_utils import Hierarchy


class HierarchyValidator:
    """
    A class to validate hierarchical data structures.
    """

    def __init__(self):
        self.catalog = HierarchyCatalog()

    def validate(self, hierarchy: Hierarchy) -> bool:
        """
        Validate the hierarchy structure.
        Returns True if valid, False otherwise.
        """
        return self._is_valid_hierarchy(hierarchy)
    
    def validate_all(self) -> dict:
        """
        Validate all hierarchies in the catalog.
        Returns a dictionary with hierarchy names as keys and validation results as values.
        """
        results = {}
        for h_name in self.catalog.hierarchies.keys():
            hierarchy = self.catalog.get(h_name)
            results[h_name] = self._is_valid_hierarchy(hierarchy)
        return results
    
    def _is_valid_hierarchy(self, hierarchy: Hierarchy) -> bool:
        tests = {}
        tests['unique_root_check'] = self._unique_root_check(hierarchy)
        tests['root_highest_id_check'] = self._root_highest_id_check(hierarchy)
        tests['tree_check'] = nx.is_tree(hierarchy.graph)
        tests['cycles_check'] = self._cycles_check(hierarchy)
        tests['dag_check'] = self._dag_check(hierarchy)
        tests['connected_check'] = self._connected_check(hierarchy)
        tests['edges_check'] = self._check_edges(hierarchy)
        return tests
        
    def _root_highest_id_check(self, hierarchy: Hierarchy) -> bool:
        root_nodes = [n for n in hierarchy.graph.nodes if hierarchy.graph.in_degree(n) == 0]
        highest_id_node = max(hierarchy.graph.nodes)
        return root_nodes[0] == highest_id_node
        
    def _unique_root_check(self, hierarchy: Hierarchy) -> bool:
        root_nodes = [n for n in hierarchy.graph.nodes if hierarchy.graph.in_degree(n) == 0]
        return len(root_nodes) == 1

    def _check_edges(self, hierarchy: Hierarchy) -> None:
        num_edges = hierarchy.number_of_edges
        num_nodes = hierarchy.number_of_nodes
        return num_edges == num_nodes - 1
    
    def _cycles_check(self, hierarchy: Hierarchy) -> bool:
        cycles = list(nx.simple_cycles(hierarchy.graph))
        if cycles:
            return False
        return True
    
    def _dag_check(self, hierarchy: Hierarchy) -> bool:
        return nx.is_directed_acyclic_graph(hierarchy.graph)
    
    def _connected_check(self, hierarchy: Hierarchy) -> bool:
        return nx.is_weakly_connected(hierarchy.graph)    

    def _check_tree(self, hierarchy: Hierarchy):
        # Check if the graph is a directed acyclic graph (DAG)
        if not nx.is_directed_acyclic_graph(hierarchy.graph):
            cycles = list(nx.simple_cycles(hierarchy.graph))
            raise ValueError(f"The graph contains cycles: {cycles}")

        num_nodes = hierarchy.number_of_nodes
        num_edges = hierarchy.number_of_edges
        print(f"Number of nodes: {num_nodes}, Number of edges: {num_edges}")

        # Check if the graph is connected
        if not nx.is_weakly_connected(hierarchy.graph):
            components = list(nx.weakly_connected_components(hierarchy.graph))
            raise ValueError(f"The graph is not connected. Components: {components}")

        # Check for multiple root nodes
        root_candidates = [n for n in hierarchy.graph.nodes if hierarchy.graph.in_degree(n) == 0]
        print(f"Root candidates: {root_candidates}")
        if not self._root_highest_id_check(hierarchy):
            print("The root is not the highest node.")
        # assert root_candidates[0] == self.graph.number_of_edges(), f"The root is not the highest node."
        if not self._unique_root_check(hierarchy):
            raise ValueError(f"The graph has {len(root_candidates)} root nodes: {root_candidates}")

        cycles = list(nx.simple_cycles(hierarchy.graph))
        if cycles != []:
            print(f"Cycles: {cycles}")
        self_loops = list(nx.selfloop_edges(hierarchy.graph))
        if self_loops != []:
            print(f"Self-loops: {self_loops}")
        duplicate_edges = [e for e in hierarchy.graph.edges if hierarchy.graph.number_of_edges(*e) > 1]
        if duplicate_edges != []:
            print(f"Duplicate edges: {duplicate_edges}")
        print(f"Is tree: {nx.is_tree(hierarchy.graph)}")
        if not self._check_edges(hierarchy):
            print("The graph does not have exactly n-1 edges for n nodes.")
            # Check for extra edges using a spanning tree
        spanning_tree = nx.minimum_spanning_tree(hierarchy.graph.to_undirected())
        extra_edges = set(hierarchy.graph.edges) - set(spanning_tree.edges)
        assert nx.is_tree(hierarchy.graph), f"The created graph is not a tree."


if __name__ == "__main__":
    validator = HierarchyValidator()
    results = validator.validate_all()
    for h_name, is_valid in results.items():
        print(f"Hierarchy '{h_name}'. Passed: {all(is_valid.values())}. Test results: {is_valid}")
        
    print()
    print("#" * 50)
    print("INVALID HIERARCHIES")
    for h_name, is_valid in results.items():
        if not all(is_valid.values()):
            print(f"Hierarchy '{h_name}' failed validation. Issues found: {is_valid}")
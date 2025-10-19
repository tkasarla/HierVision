import json
import networkx as nx

import os
from glob import iglob


class Hierarchy:
    """
    A class to represent and manipulate a hierarchical structure defined in a JSON file.
    Args:
        hierarchy_file (str): Path to the JSON file containing the hierarchy definition.
    Attributes:
        graph (nx.DiGraph): A directed graph representing the hierarchy.
    Methods:
        load_hierarchy_file(file_path, throw_error=True): Class method to load a hierarchy from a JSON file.
        number_of_nodes: Property to get the number of nodes in the hierarchy.
        depth: Property to get the depth of the hierarchy.
        number_of_leaves: Property to get the number of leaf nodes in the hierarchy.
        average_branching_factor: Property to get the average branching factor of the hierarchy.
    """
    
    def __init__(self, hierarchy_file):
        self.hierarchy_file = hierarchy_file
        self.graph, self.root_id = self.load_hierarchy_file(hierarchy_file)

    @classmethod
    def load_hierarchy_file(cls, file_path: str, throw_error=True) -> nx.DiGraph:
        """
        Load a hierarchy from a JSON file and create a directed graph.
        Args:
            file_path (str): Path to the JSON file containing the hierarchy definition.
            throw_error (bool): Whether to raise an error if the hierarchy is invalid.
        Returns:
            nx.DiGraph: A directed graph representing the hierarchy.
        """
        
        with open(file_path, "r") as json_file:
            graph_dict = json.load(json_file)

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes
        for node in graph_dict["nodes"]:
            G.add_node(node["id"], label=node["label"])

        # Add edges
        for link in graph_dict["links"]:
            G.add_edge(link["source"], link["target"])

        root_id = [node for node, degree in G.in_degree() if degree == 0]

        if throw_error and len(root_id) != 1:
            print(f"Warning: There should be exactly one root node in the hierarchy {file_path}. Found {len(root_id)}")
            # raise ValueError(f"There should be exactly one root node in the hierarchy {file_path}. Found {len(root_id)}")
        else:
            if any([isinstance(s, str) for s in root_id]):
                root_id = root_id[0]  # If root_id is a list with one string, take that string
            else:
                root_id = max(root_id)

        return G, root_id

    @property
    def number_of_nodes(self):
        """
        Get the number of nodes in the hierarchy.
        Returns:
            int: Number of nodes in the hierarchy.
        """
        if not hasattr(self, '_number_of_nodes'):
            self._number_of_nodes = self.graph.number_of_nodes()
        return self._number_of_nodes
    
    @property
    def number_of_edges(self):
        """
        Get the number of edges in the hierarchy.
        Returns:
            int: Number of edges in the hierarchy.
        """
        if not hasattr(self, '_number_of_edges'):
            self._number_of_edges = self.graph.number_of_edges()
        return self._number_of_edges

    @property
    def depth(self):
        """
        Get the depth of the hierarchy.
        Returns:
            int: Depth of the hierarchy.
        """
        if not hasattr(self, '_depth'):
            self._depth = max(nx.single_source_shortest_path_length(self.graph, self.root_id).values())
        return self._depth

    @property
    def number_of_leaves(self):
        """
        Get the number of leaf nodes in the hierarchy.
        Returns:
            int: Number of leaf nodes in the hierarchy.
        """
        if not hasattr(self, '_number_of_leaves'):
            self._number_of_leaves = len([n for n in self.graph.nodes if self.graph.out_degree(n) == 0])
        return self._number_of_leaves
    
    @property
    def number_of_classes(self):
        """
        Get the number of classes (leaf nodes) in the hierarchy. Alias for `number_of_leaves`.
        Returns:
            int: Number of classes in the hierarchy.
        """
        return self.number_of_leaves

    @property
    def average_branching_factor(self):
        """
        Get the average branching factor of the hierarchy.
        Returns:
            float: Average branching factor of the hierarchy.
        Explanation: 
        The average branching factor is calculated as the average number of children per internal node (meaning excluding leaf nodes).
        """
        if not hasattr(self, '_average_branching_factor'):
            if self.number_of_nodes == 0:
                self._average_branching_factor = 0.0
            total_children = sum(self.graph.out_degree(n) for n in self.graph.nodes if self.graph.out_degree(n) > 0)
            internal_nodes_count = sum(1 for n in self.graph.nodes if self.graph.out_degree(n) > 0)
            if internal_nodes_count == 0:
                self._average_branching_factor = 0.0
            else:
                self._average_branching_factor = total_children / internal_nodes_count
        return self._average_branching_factor
    
    def summary(self):
        """
        Print a summary of the hierarchy's properties.
        """
        print(self.__repr__())
        
    def __repr__(self):
        out = f"Hierarchy Summary:\n"
        out += f"Number of nodes: {self.number_of_nodes}\n"
        out += f"Number of edges: {self.number_of_edges}\n"
        out += f"Depth: {self.depth}\n"
        out += f"Number of leaves: {self.number_of_leaves}\n"
        out += f"Average branching factor: {self.average_branching_factor:.2f}\n"
        return out
    
    
class HierarchyCatalog:
    """
    A catalog to manage and access different hierarchy files for datasets.
    
    Args:
        base_path (str): Base directory where hierarchy files are stored.
        pattern (str): Pattern to locate hierarchy files within the base directory.
    Attributes:
        hierarchies (dict): A dictionary mapping dataset names to their hierarchy file paths.
    Methods:
        __getitem__(dataset_name): Retrieve the Hierarchy object for a given dataset name.
        get(dataset_name): Alternative method to retrieve the Hierarchy object for a given dataset name.
    """
    def __init__(self, base_path: str='./hierarchies', pattern: str='**/{dataset_name}/**/hierarchy.json'):
        self.base_path = base_path
        self.pattern = pattern
        self.hierarchies = self._init_catalog()
            
    def __getitem__(self, dataset_name: str) -> str:
        try:
            hierarchy = Hierarchy(self.hierarchies[dataset_name])
            return hierarchy
        except KeyError:
            raise ValueError(f"Dataset {dataset_name} not found in catalog.")

    def get(self, dataset_name: str) -> str:
        return self[dataset_name]
        
    def _init_catalog(self):
        path_to_hierarchies = os.path.join(self.base_path, self.pattern)
        dataset_names = os.listdir(self.base_path)
    
        datasets = {}
        for dataset_folder in dataset_names:
            file_path = path_to_hierarchies.format(dataset_name=dataset_folder)
            file_path = iglob(file_path, recursive=True)  # Get the first matching file
            for path in file_path:
                dataset_key = os.path.dirname(path).replace(self.base_path, '').replace(os.sep, '-').strip('-')
                datasets[dataset_key] = path
        return datasets
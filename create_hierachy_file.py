import os 
import argparse
import json
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import kaleido
import warnings
from loguru import logger
from glob import glob

warnings.filterwarnings("ignore")


class BitMatrix:
    def __init__(self, size):
        self.size = size
        self.rows = size[0]
        self.cols = (size[1] // 64) + (size[1] % 64 != 0)
        self._data = np.zeros((self.rows, self.cols), dtype=np.uint64)

    def __getitem__(self, *idxs):
        row, col = idxs
        if row < 0 or row >= self.rows or col < 0 or col >= self.size[1]:
            raise IndexError("Index out of bounds")
        return (self._data[row, col // 64] >> (col % 64)) & 1

    def __setitem__(self, idxs, value):
        row, col = idxs
        if row < 0 or row >= self.rows or col < 0 or col >= self.size[1]:
            raise IndexError("Index out of bounds")
        word_idx = col // 64
        bit = col % 64
        if value:
            self._data[row, word_idx] |= (1 << bit)
        else:
            self._data[row, word_idx] &= ~(1 << bit)

    def sum(self, axis=None):
        if axis is None:
            raise NotImplementedError
        elif axis == 0:
            raise NotImplementedError
        elif axis == 1:
            cols_sum = np.zeros(self.rows, dtype=np.uint64)
            for row in range(self.rows):
                cols_sum[row] = np.sum(np.bitwise_count(self._data[row]))
            return cols_sum
        else:
            raise ValueError("Invalid axis. Use 0 or 1.")


class HierarchyFileCreator:

    datasets = {
        "paco",
        "mapillary",
        "ade20k_wordnet",
        "ade20k_coarse_to_fine",
        "ade20k_scene_cls_train",
        "ade20k_scene_cls_val",
        "biotrove-balanced",
        "biotrove-lifestages",
        "biotrove-unseen",
        "rare_species",
        "tree_of_life",
        "imagenet-ood",
    }

    format_template = {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [
            { "id": "root", "label": "Root Node" },
            { "id": "child1", "label": "Child 1" },
            ...
        ],
        "links": [
            { "source": "root", "target": "child1" },
            ...
        ]
    }

    def __init__(self, dataset_name: str, dataset_path: str, hierarchy_file_path: str):
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} is not supported.")
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_dir = os.path.basename(dataset_path)
        self.hierarchy_files_path = os.path.join(hierarchy_file_path, self.dataset_dir)
        os.makedirs(self.hierarchy_files_path, exist_ok=True)
        self._root_label = "root"

    def _ade20k_wordnet(self):
        ds_metadata = pd.read_pickle(os.path.join(self.dataset_path, "index_ade20k.pkl"))

        chains = ds_metadata["wordnet_synset"]
        G = nx.DiGraph()
        for entry in chains:
            hierarchy = entry.split(".")[::-1]
            for j in range(len(hierarchy) - 1):
                parent, child = hierarchy[j].strip(), hierarchy[j + 1].strip()
                G.add_node(parent, label=parent)
                G.add_node(child, label=child)
                if not G.has_edge(parent, child):
                    G.add_edge(parent, child)

        logger.info(f"Created graph for {self.dataset_name} with {G.number_of_nodes()} nodes.")
        return G
    
    def _ade20k_coarse_to_fine(self):
        annotation_jsons = glob(os.path.join(self.dataset_path, "images", "**", "*.json"), recursive=True)
        G = nx.DiGraph()
        for json_file in annotation_jsons:
            with open(json_file, 'r', encoding="ISO-8859-1", errors="ignore") as f:
                content = f.read()
            
            content = content.encode("utf-8", errors="ignore")
            objects = json.loads(content)["annotation"]["object"]
            id_to_name = {}
            part_of = {}
            for obj in objects:
                node = obj["name"]
                id_to_name[obj["id"]] = node
                part_of[node] = obj["parts"]["ispartof"]
                if node not in G:
                    G.add_node(node, label=node)
            
            for node in part_of:
                parent_id = part_of[node]
                if parent_id != [] and parent_id in id_to_name:
                    parent = id_to_name[parent_id]
                    if parent not in G:
                        G.add_node(parent, label=parent)
                    if not G.has_edge(parent, node):
                        G.add_edge(parent, node)
        logger.info(f"Created graph for {self.dataset_name} with {G.number_of_nodes()} nodes.")
        return G
    
    def _ade20k_scene_cls(self, split: str):
        folders = os.path.join(self.dataset_path, "images", "ADE", split)
        folders = glob(os.path.join(folders, "*", "*"))
        folders = [f for f in folders if os.path.isdir(f)]

        G = nx.DiGraph()
        for folder in folders:
            fine_label = os.path.basename(folder)
            coarse_label = os.path.basename(os.path.dirname(folder))
            if not G.has_node(coarse_label):
                G.add_node(coarse_label, label=coarse_label)
            if not G.has_node(fine_label):
                G.add_node(fine_label, label=fine_label)
            if not G.has_edge(coarse_label, fine_label):
                G.add_edge(coarse_label, fine_label)
        logger.info(f"Created graph for {self.dataset_name}/{split} with {G.number_of_nodes()} nodes.")
        return G

    def _mapillary(self):
        with open(os.path.join(self.dataset_path, "config.json"), 'r') as f:
            annotation = json.load(f)
        
        G = nx.DiGraph()
        for label in annotation["labels"]:
            node = label["readable"]
            hierarchy = label["name"].split("--")
            if not G.has_node(node):
                G.add_node(node, label=node)
            if len(hierarchy) > 1:
                hierarchy = hierarchy[:-1] + [node]
                for i in range(len(hierarchy) - 1):
                    parent, child = hierarchy[i], hierarchy[i + 1]
                    if not G.has_node(parent):
                        G.add_node(parent, label=parent)
                    if not G.has_node(child):
                        G.add_node(child, label=child)
                    if not G.has_edge(parent, child):
                        G.add_edge(parent, child)
        logger.info(f"Created graph for {self.dataset_name} with {G.number_of_nodes()} nodes.")
        return G

    def _biotrove(self, split: str):
        cols = ["kingdom","phylum","class","order","family","genus","species"]
        chains = pd.read_csv(os.path.join(self.dataset_path, "BioTrove-benchmark", "downloaded", f"BioTrove-{split}.csv"), sep=",", header="infer", usecols=cols)

        G = nx.DiGraph()
        for _, row in chains.iterrows():
            hierarchy = [row[col] for col in cols if pd.notna(row[col])]
            for i in range(len(hierarchy) - 1):
                parent, child = hierarchy[i], hierarchy[i + 1]
                G.add_node(parent, label=parent)
                G.add_node(child, label=child)
                if not G.has_edge(parent, child):
                    G.add_edge(parent, child)

        logger.info(f"Created graph for {self.dataset_name}/{split} with {G.number_of_nodes()} nodes.")
        return G
    
    def _rare_species(self):
        cols = ["kingdom","phylum","class","order","family","genus","species"]
        chains = pd.read_csv(os.path.join(self.dataset_path, "metadata", "rarespecies-catalog.csv"), sep=",", header="infer", usecols=cols)

        G = nx.DiGraph()
        for _, row in chains.iterrows():
            hierarchy = [row[col] for col in cols if pd.notna(row[col])]
            for i in range(len(hierarchy) - 1):
                parent, child = hierarchy[i], hierarchy[i + 1]
                G.add_node(parent, label=parent)
                G.add_node(child, label=child)
                if not G.has_edge(parent, child):
                    G.add_edge(parent, child)

        logger.info(f"Created graph for {self.dataset_name}/rare_species with {G.number_of_nodes()} nodes.")
        return G

    def _tree_of_life(self):
        hierarchy_ids_column = "hierarchy_string_tsn"
        hierarchy_column = "hierarchy_string_names"

        chains = pd.read_csv(os.path.join(self.dataset_path, "metadata/species_level_taxonomy_chains.csv"), usecols=[hierarchy_ids_column, hierarchy_column], header=0)
        G = nx.DiGraph()
        for _, row in chains.iterrows():
            hierarchy = row[hierarchy_column].split("->")
            hierarchy_ids = row[hierarchy_ids_column].split("-")
            for i in range(len(hierarchy) - 1):
                parent, pid = hierarchy[i], hierarchy_ids[i]
                child, cid = hierarchy[i + 1], hierarchy_ids[i + 1]
                G.add_node(pid, label=parent)
                G.add_node(cid, label=child)
                if not G.has_edge(pid, cid):
                    G.add_edge(pid, cid)

        logger.info(f"Created graph for {self.dataset_name} with {G.number_of_nodes()} nodes.")
        return G
    
    def _assign_level_ids(self, G):
        logger.info("Assigning level ids to nodes based on their hierarchy levels.")

        # Map to store assigned integer ids
        assigned_ids = {}
        next_id = 0

        # Step 1: Find all leaves (nodes with no outgoing edges)
        logger.info("Identifying leaf nodes in the graph.")
        leaves = [n for n in G.nodes if G.out_degree(n) == 0]

        # Step 2: Assign IDs to leaves
        logger.info(f"Assigning IDs to {len(leaves)} leaf nodes.")
        for leaf in leaves:
            assigned_ids[leaf] = next_id
            next_id += 1

        # Step 3: Assign IDs to internal nodes
        logger.info("Processing internal nodes.")
        # For cyclic graphs, use a queue to process nodes whose children are already assigned
        unassigned = set(G.nodes) - set(assigned_ids)
        while unassigned:
            progress = False
            for node in list(unassigned):
                children = list(G.successors(node))
                # If all children are assigned or there are no children (cycle), assign id
                if all(child in assigned_ids for child in children) or not children:
                    assigned_ids[node] = next_id
                    next_id += 1
                    unassigned.remove(node)
                    progress = True
            if not progress:
                # Assign IDs to remaining nodes (in cycles) arbitrarily
                for node in unassigned:
                    assigned_ids[node] = next_id
                    next_id += 1
                break

        G = nx.relabel_nodes(G, assigned_ids, copy=True)

        logger.info(f"Assigned {len(assigned_ids)} unique ids to nodes in the graph.")
        return G, len(leaves)
    
    def _add_virtual_root(self, G):
        logger.info("Adding virtual root node to the graph.")
        # Find nodes with no incoming edges
        roots = [n for n in G.nodes if G.in_degree(n) == 0]

        if len(roots) == 1:
            logger.info("Only one root candidate found, no need to add a virtual root.")
            return G

        # Add a virtual root node
        G.add_node("root", label=self._root_label)

        # Connect root to all current root candidates
        for node in roots:
            G.add_edge("root", node)
        logger.info(f"Virtual root added with {G.number_of_nodes()} total nodes.")
        return G
    
    def _tree_depth(self, G, root):
        lengths = nx.single_source_shortest_path_length(G, root)
        return max(lengths.values())

    def _extract_subtree(self, G, root, max_depth, subsample=1.0):
        if root not in G:
            raise ValueError(f"Node '{root}' not in graph")

        visited = set()
        queue = [(root, 0)]
        subtree_nodes = []

        pruning_factors = np.linspace(1.0, subsample, num=max_depth + 1)

        while queue:
            node, depth = queue.pop(0)
            if node in visited or depth > max_depth:
                continue
            if np.random.rand() < pruning_factors[depth]: # Randomly sample nodes to reduce size
                visited.add(node)
                subtree_nodes.append(node)
                for child in G.successors(node):
                    queue.append((child, depth + 1))

        return G.subgraph(subtree_nodes).copy()
    
    def _visualize_hierarchy(self, G, number_of_nodes=None, max_depth=3):
        # Compute radial layout
        def radial_layout(G, root=0, depth=3):
            pos = {root: (0, 0)}  # root at center
            layers = [[] for _ in range(depth + 1)]
            layers[0].append(root)

            # BFS to organize nodes by level
            queue = [(root, 0)]
            while queue:
                node, level = queue.pop(0)
                if level < depth:
                    children = list(G.successors(node))
                    layers[level + 1].extend(children)
                    for child in children:
                        queue.append((child, level + 1))

            # Assign positions in concentric circles
            for level in range(1, depth + 1):
                radius = level  # radial distance can be scaled
                nodes = layers[level]
                angle_step = 2 * math.pi / len(nodes)
                for i, node in enumerate(nodes):
                    theta = i * angle_step
                    x = radius * math.cos(theta)
                    y = radius * math.sin(theta)
                    pos[node] = (x, y)

            return pos

        root = number_of_nodes - 1
        G_sub = self._extract_subtree(G, root, max_depth=max_depth, subsample=1/(math.floor(np.log10(number_of_nodes))))
        max_depth = self._tree_depth(G_sub, root)
        pos = radial_layout(G_sub, root=root, depth=max_depth)

        # Draw the graph
        plt.figure(figsize=(8, 8))
        node_colors = ['orange' if node == root else 'deepskyblue' for node in G_sub.nodes()]
        nx.draw(G_sub, pos, with_labels=False, arrows=True,
                node_size=7, node_color=node_colors, edge_color='darkgray')
        plt.axis('off')
        plt.savefig(os.path.join(self.hierarchy_files_path, "hierarchy.png"), bbox_inches='tight')
        plt.close()

    def _visualize_hierarchy_from_file(self, json_file):
        # Load the JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Create a directed graph
        G = nx.DiGraph()
        logger.info(f"Loaded graph and data")

        # Add nodes
        for node in data["nodes"]:
            G.add_node(node["id"], label=node["label"])

        # Add edges
        for link in data["links"]:
            G.add_edge(link["source"], link["target"])

        logger.info(f"added {len(G.nodes())} nodes and {len(G.edges())} edges to the graph")
        # Get labels for nodes
        labels = nx.get_node_attributes(G, 'label')
        logger.info(f" got {len(labels)} labels for the nodes")

        # Generate a radial tree using Plotly's Sunburst chart
        ids = []
        labels_list = []
        parents = []

        for node in G.nodes():
            ids.append(node)
            labels_list.append(labels[node])
            # Find the parent of the node
            parent = next(iter(G.predecessors(node)), "")  # Root node has no parent
            parents.append(parent)

        # Create the Sunburst chart
        fig = go.Figure(go.Sunburst(
            ids=ids,
            labels=labels_list,
            parents=parents,
            branchvalues="total"
        ))

        fig.update_layout(
            margin=dict(t=0, l=0, r=0, b=0)
        )

        fig.show()
        # save as png
        fig.write_image(os.path.join(self.hierarchy_files_path, "hierarchy_kaleido.png"))

    def create_hierarchy_file(self):
        logger.info(f"Creating hierarchy file for dataset: {self.dataset_name}")
        if self.dataset_name == "tree_of_life":
            G = self._tree_of_life()
        elif "ade20k" in self.dataset_name:
            if "wordnet" in self.dataset_name:
                logger.info("Creating hierarchy for ADE20K using WordNet.")
                G = self._ade20k_wordnet()
            elif "coarse_to_fine" in self.dataset_name:
                logger.info("Creating hierarchy for ADE20K using coarse-to-fine.")
                G = self._ade20k_coarse_to_fine()
            elif "scene_cls" in self.dataset_name:
                split = self.dataset_name.split("_")[-1]
                split = "training" if split == "train" else "validation"
                logger.info(f"Creating hierarchy for ADE20K scene classification ({split}).")
                G = self._ade20k_scene_cls(split)
        elif self.dataset_name == "mapillary":
            G = self._mapillary()
        elif "biotrove" in self.dataset_name:
            if "balanced" in self.dataset_name:
                split = "Balanced"
            elif "lifestages" in self.dataset_name:
                split = "LifeStages"
            elif "unseen" in self.dataset_name:
                split = "Unseen"
            G = self._biotrove(split)
        elif self.dataset_name == "rare_species":
            G = self._rare_species()
        elif self.dataset_name == "imagenet-ood":
            raise NotImplementedError("Hierarchy creation for imagenet-ood is not implemented.")
        elif self.dataset_name == "paco":
            raise NotImplementedError("Hierarchy creation for PACO is not implemented.")
        else:
            raise NotImplementedError(f"Hierarchy creation for {self.dataset_name} is not implemented.")

        # Add virtual root node if it doesn't exist
        G = self._add_virtual_root(G)

        # Assign level ids
        G, number_of_classes = self._assign_level_ids(G)

        # Export in JSON format
        logger.info("Exporting hierarchy to JSON format.")
        with open(os.path.join(self.hierarchy_files_path, "hierarchy.json"), 'w') as f:
            data = nx.node_link_data(G, edges="links")
            data['nodes'] = sorted(data['nodes'], key=lambda x: x['id'])
            json.dump(data, f, indent=4)

        # Export dataset details in Markdown format
        logger.info("Exporting dataset details in Markdown format.")
        number_of_nodes = G.number_of_nodes()
        max_depth = self._tree_depth(G, number_of_nodes - 1)
        dataset_details = f"""
        {self.dataset_name.replace("_", " ").capitalize()}
        number of levels in hierarchy: {max_depth}
        number of classes: {number_of_classes}
        """

        with open(os.path.join(self.hierarchy_files_path, "dataset_details.md"), 'w') as f:
            f.write(dataset_details)

        # Visualize the hierarchy
        logger.info("Visualizing the hierarchy.")
        self._visualize_hierarchy(G, number_of_nodes, max_depth)
        self._visualize_hierarchy_from_file(os.path.join(self.hierarchy_files_path, "hierarchy.json"))


def main():
    parser = argparse.ArgumentParser(description="Create hierarchy file for a dataset.")
    parser.add_argument("dataset_name", type=str, choices=HierarchyFileCreator.datasets, help="Name of the dataset")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("hierarchy_file_path", type=str, help="Path to save the hierarchy file")

    args = parser.parse_args()

    creator = HierarchyFileCreator(args.dataset_name, args.dataset_path, args.hierarchy_file_path)
    creator.create_hierarchy_file()
    logger.info(f"Hierarchy file created at {args.hierarchy_file_path}")


if __name__ == "__main__":
    main()
    # Run the script from command line
    # python create_hierachy_file.py tree_of_life /path/to/dataset /path/to/hierarchy_file.json


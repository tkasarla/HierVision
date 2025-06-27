import argparse
import json
import math
import os
import warnings
from glob import glob

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from collections import defaultdict
from loguru import logger
from nltk.corpus import wordnet as wn
from typing import List, Set

from utils.graph_utils import load_graph_from_file

warnings.filterwarnings("ignore")



class HierarchyFileCreator:

    # Supported datasets
    datasets = {
        "ade20k_coarse_to_fine", # semantic segmentation
        "ade20k_scene_cls-train", # semantic segmentation
        "ade20k_scene_cls-val", # semantic segmentation
        "ade20k_wordnet", # semantic segmentation
        "babel", # action recognition
        "biotrove-balanced", # image classification
        "biotrove-lifestages", # image classification
        "biotrove-unseen", # image classification
        "cityscapes", # semantic segmentation
        "imagenet21k", # image classification
        "imagenetood", # image classification
        "mapillary", # semantic segmentation
        "objects365", # object detection
        "paco-ego4d-test", # semantic segmentation
        "paco-ego4d-train", # semantic segmentation
        "paco-ego4d-val", # semantic segmentation
        "paco-lvis-test", # semantic segmentation
        "paco-lvis-train", # semantic segmentation
        "paco-lvis-val", # semantic segmentation
        "rare_species", # image classification
        "tree_of_life", # image classification
        "visual_genome", # image classification
    }

    # Template for the hierarchy file format (example structure)
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

    def __init__(self, dataset_name: str, dataset_path: str, hierarchy_file_path: str, root_label: str = "root"):
        """
        This class creates a hierarchy file for a given dataset using `networkx`.

        Args:
            dataset_name (str): Name of the dataset. Must be one of the supported datasets.
            dataset_path (str): Path to the dataset directory.
            hierarchy_file_path (str): Path to save the hierarchy file.
            root_label (str): Label for the root node in the hierarchy. Default is "root".
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset {dataset_name} is not supported.")
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.dataset_dir = os.path.basename(dataset_path)
        self.hierarchy_files_path = os.path.join(hierarchy_file_path, self.dataset_dir)
        os.makedirs(self.hierarchy_files_path, exist_ok=True)
        self._root_label = root_label

    ### Methods to create hierarchy graphs for different datasets

    def _ade20k_coarse_to_fine(self):
        """
        Creates a hierarchy graph for the ADE20K dataset using coarse-to-fine annotations.
        The hierarchy is constructed based on the 'ispartof' relationships in the annotations.
        
        Args:
            None
        Returns:
            G (nx.DiGraph): Directed graph representing the hierarchy.
        """
        annotation_jsons = glob(os.path.join(self.dataset_path, "images", "**", "*.json"), recursive=True)
        G = nx.DiGraph()
        for json_file in annotation_jsons:
            with open(json_file, 'r', encoding="ISO-8859-1", errors="ignore") as f:
                content = f.read()
            
            content = content.encode("utf-8", errors="ignore")
            objects = json.loads(content)["annotation"]["object"]
            id_to_name = {}
            part_of = {}
            local_to_global_ids = {}

            for obj in objects:
                obj_id = obj["id"]
                node = obj["raw_name"]
                if node == "root":
                    node = node + " (entity)" # to avoid conflicts with the root node
                node_id = obj["name_ndx"]
                local_to_global_ids[obj_id] = node_id
                id_to_name[node_id] = node
                part_of[node_id] = obj["parts"]["ispartof"]
                if node_id not in G:
                    G.add_node(node_id, label=node)
            try:
                for node_id in part_of:
                    if (local_parent_id := part_of[node_id]) == []: 
                        continue
                    if local_parent_id not in local_to_global_ids:
                        continue
                    parent_id = local_to_global_ids[local_parent_id]
                    # Fix 1: Check for valid parent and prevent self-loops
                    if (parent_id in id_to_name and 
                        parent_id != node_id):  # Prevent self-loops
                        
                        parent = id_to_name[parent_id]
                        # Fix 2: Check parent_id instead of parent string
                        if parent_id not in G:
                            G.add_node(parent_id, label=parent)
                        
                        # Fix 3: Prevent cycles by checking if adding edge would create cycle
                        if not G.has_edge(parent_id, node_id) and not nx.has_path(G, node_id, parent_id):
                            G.add_edge(parent_id, node_id)
                        elif nx.has_path(G, node_id, parent_id):
                            logger.warning(f"Skipping edge {parent_id}->{node_id} to prevent cycle")
            except Exception as e:
                print(e)
                breakpoint()
    
        logger.info(f"Created graph for {self.dataset_name} with {G.number_of_nodes()} nodes.")
        return G
    
    def _ade20k_scene_cls(self, split: str):
        """
        Creates a hierarchy graph for the ADE20K dataset for scene classification tasks.
        The hierarchy is inferred from the dataset's folder structure.
        
        Args:
            split (str): Split name, either "training" or "validation".
        Returns:
            G (nx.DiGraph): Directed graph representing the hierarchy.
        """
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

    def _ade20k_wordnet(self):
        """
        Creates a hierarchy graph for the ADE20K dataset using WordNet synsets.
        The hierarchy is constructed based on the WordNet synsets provided in the dataset metadata.
        
        Args:
            None
        Returns:
            G (nx.DiGraph): Directed graph representing the hierarchy.
        """
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

    def _babel(self):
        """
        Creates a hierarchy graph for the Babel dataset.
        TODO: Extract the hierarchy from the annotation files or from https://babel.is.tue.mpg.de/media/upload/stats/babel_label_org.html.
        """
        pass
    
    def _biotrove(self, split: str):
        """
        Creates a hierarchy graph for the BioTrove dataset.
        The hierarchy is constructed based on the taxonomic chains provided in the dataset.

        Args:
            split (str): Split name, either "Balanced", "LifeStages", or "Unseen".
        Returns:
            G (nx.DiGraph): Directed graph representing the hierarchy.
        """
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
    
    def _cityscapes(self):
        """
        Creates a hierarchy graph for the Cityscapes dataset.
        The hierarchy is constructed based on the class definitions extracted from https://www.cityscapes-dataset.com/dataset-overview/.
        
        Args:
            None
        Returns:
            G (nx.DiGraph): Directed graph representing the hierarchy.
        """
        with open(os.path.join(self.dataset_path, "class_definitions.json"), "r") as f:
            hierarchy_data = json.load(f)
        
        G = nx.DiGraph()
        for i, sup_cat in enumerate(hierarchy_data):
            G.add_node(i, label=sup_cat)
            for sub_cat in hierarchy_data[sup_cat]:
                G.add_node(sub_cat, label=sub_cat)
                if not G.has_edge(i, sub_cat):
                    G.add_edge(i, sub_cat)
        logger.info(f"Created graph for {self.dataset_name} with {G.number_of_nodes()} nodes.")
        return G

    def _imagenet21k(self):
        """
        Creates a hierarchy graph for the ImageNet-21K dataset.
        The hierarchy is constructed based on the WordNet synsets of the ImageNet classes.

        Args:
            None
        Returns:
            G (nx.DiGraph): Directed graph representing the hierarchy.
        """
        # child_to_parent, id_to_name = self._get_imagenet21k_mapping()

        # G = nx.DiGraph()
        # for child, parent in child_to_parent.items():
        #     child_name = id_to_name[child]
        #     parent_name = id_to_name[parent]
        #     G.add_node(child, label=child_name)
        #     G.add_node(parent, label=parent_name)
        #     if not G.has_edge(parent, child):
        #         G.add_edge(parent, child)
        class_list = self._get_imagenet21k_mapping()
        class_list = [int(class_id[1:]) for class_id in class_list]  # Convert to integer IDs
        G = self._apply_wordnet_hierarchy(entity_ids=class_list)

        logger.info(f"Created graph for {self.dataset_name} with {G.number_of_nodes()} nodes.")
        return G

    def _imagenetood(self):
        """
        Creates a hierarchy graph for the ImageNet-OOD dataset.
        The hierarchy is constructed based on the WordNet synsets of the ImageNet classes.

        Args:
            None
        Returns:
            G (nx.DiGraph): Directed graph representing the hierarchy.
        """
        images = glob(os.path.join(self.dataset_path, "*.JPEG"))
        class_ids = set((int(os.path.basename(img).split("_")[0][1:]) for img in images))
        # imagenet_classes = self._imagenet_id_to_name()
        # child_to_parent, id_to_name = self._get_imagenet21k_mapping()

        # G = nx.DiGraph()
        # for class_id in class_ids:
        #     child_name = id_to_name.get(class_id, imagenet_classes[class_id])
        #     G.add_node(class_id, label=child_name)

        #     if class_id in child_to_parent:
        #         parent_name = id_to_name[child_to_parent[class_id]]
        #         G.add_node(parent_name, label=parent_name)
        #         if not G.has_edge(parent_name, class_id):
        #             G.add_edge(parent_name, class_id)
        G = self._apply_wordnet_hierarchy(entity_ids=class_ids)
        logger.info(f"Created WordNet graph for {self.dataset_name} with {G.number_of_nodes()} nodes.")
        return G

    def _mapillary(self):
        """
        Creates a hierarchy graph for the Mapillary Vistas dataset.
        The hierarchy is constructed based on the class definitions within the annotation files.
        
        Args:
            None
        Returns:
            G (nx.DiGraph): Directed graph representing the hierarchy.
        """
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

    def _objects365(self):
        """
        Creates a hierarchy graph for the Objects365 dataset.
        The hierarchy is constructed based on the scraped information from https://www.objects365.org/explore.html.

        Args:
            None
        Returns:
            G (nx.DiGraph): Directed graph representing the hierarchy.
        """
        with open(os.path.join(self.dataset_path, "objects365_hierarchy.json"), "r") as f:
            hierarchy_data = json.load(f)
        
        G = nx.DiGraph()
        for sup_cat in hierarchy_data:
            G.add_node(sup_cat, label=sup_cat)
            for sub_cat in hierarchy_data[sup_cat]:
                G.add_node(sub_cat, label=sub_cat)
                if not G.has_edge(sup_cat, sub_cat):
                    G.add_edge(sup_cat, sub_cat)
        logger.info(f"Created graph for {self.dataset_name} with {G.number_of_nodes()} nodes.")
        return G
    
    def _paco(self, subdataset: str, split: str):
        """
        Creates a hierarchy graph for the PACO dataset.
        The hierarchy is constructed based on the object and part annotations in the dataset.
        
        Args:
            subdataset (str): Subdataset name, either "ego4d" or "lvis".
            split (str): Split name, either "train", "val", or "test".
        Returns:
            G (nx.DiGraph): Directed graph representing the hierarchy.
        """
        # the following function is taken from the PACO repository
        def get_obj_and_part_anns(annotations):
            """
            Returns a map between an object annotation ID and 
            (object annotation, list of part annotations) pair.
            """
            obj_ann_id_to_anns = {ann["id"]: (ann, []) for ann in annotations if ann["id"] == ann["obj_ann_id"]}
            for ann in annotations:
                if ann["id"] != ann["obj_ann_id"]:
                    obj_ann_id_to_anns[ann["obj_ann_id"]][1].append(ann)
            return obj_ann_id_to_anns
        
        assert subdataset in ["ego4d", "lvis"], f"Subdataset {subdataset} is not supported."
        
        with open(os.path.join(self.dataset_path, "annotations", f"paco_{subdataset}_v1_{split}.json"), 'r') as f:
            ds_annotations = json.load(f)

        # create mappings for whole-part relationships
        cat_id_to_name = {d["id"]: d["name"] for d in ds_annotations["categories"]}
        obj_ann_id_to_anns = get_obj_and_part_anns(ds_annotations["annotations"])

        # create a directed graph
        G = nx.DiGraph()
        for ann in ds_annotations["annotations"]:
            anns = obj_ann_id_to_anns[ann["obj_ann_id"]] # (object annotation, list of part annotations)
            parent_id = anns[0]["category_id"]
            parent = cat_id_to_name[parent_id]
            G.add_node(parent_id, label=parent)
            for part_ann in anns[1]:
                child_id = part_ann["category_id"]
                child = cat_id_to_name[child_id]
                G.add_node(child_id, label=child)
                if not G.has_edge(parent_id, child_id):
                    G.add_edge(parent_id, child_id)
        logger.info(f"Created graph for {self.dataset_name}/{subdataset}/{split} with {G.number_of_nodes()} nodes.")
        return G
    
    def _rare_species(self):
        """
        Creates a hierarchy graph for the Rare Species dataset.
        The hierarchy is constructed based on the taxonomic chains provided in the dataset's catalog.

        Args:
            None
        Returns:
            G (nx.DiGraph): Directed graph representing the hierarchy.
        """
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
        """
        Creates a hierarchy graph for the Tree of Life dataset.
        The hierarchy is constructed based on the taxonomic chains provided in the dataset's metadata.
        
        Args:
            None
        Returns:
            G (nx.DiGraph): Directed graph representing the hierarchy.
        """
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

    def _visual_genome(self):
        """
        Creates a hierarchy graph for the Visual Genome dataset.
        The hierarchy is constructed based on the synsets of objects in the dataset.

        Args:
            None
        Returns:
            G (nx.DiGraph): Directed graph representing the hierarchy.
        """
        # Load Visual Genome objects data
        with open(os.path.join(self.dataset_path, "objects.json"), 'r') as f:
            objects = json.load(f)

        # Extract synset names
        synset_names = set()
        for img in objects:
            
            for obj in img["objects"]:
                synset = obj.get('synsets')
                if synset:
                    synset_names.update(synset)

        logger.info(f"Extracted {len(synset_names)} synset names from Visual Genome.")
        
        # Create WordNet hierarchy graph
        G = self._apply_wordnet_hierarchy(synset_names)

        logger.info(f"Created WordNet graph for Visual Genome with {G.number_of_nodes()} nodes.")
        return G
    
    ### (End) Methods to create hierarchy graphs for different datasets

    ### Helper methods
        
    def _get_imagenet21k_mapping(self):
        """
        Loads the ImageNet-21K class mapping from a precomputed file.
        Returns:
            class_list (list): List of ImageNet-21K class IDs.
        """
        class_mapping = torch.load(os.path.join("datasets", "imagenet21k_miil_tree.pth"), map_location="cpu", weights_only=False)
        return class_mapping["class_list"]
        # child_to_parent = class_mapping["child_2_parent"]
        # id_to_name = class_mapping["class_description"]

        # return child_to_parent, id_to_name
    
    def _imagenet_id_to_name(self):
        """
        Loads the ImageNet class index mapping from a JSON file.
        Returns:
            id_to_name (dict): Mapping from ImageNet class IDs to class names.
        """
        with open(os.path.join(os.path.dirname(self.dataset_path), "ImageNet", "imagenet_class_index.json"), "r") as f:
            imagenet_classes = json.load(f)

        id_to_name = dict(imagenet_classes.items())
        return id_to_name

    def _apply_wordnet_hierarchy(self, entity_names: Set | List = None, entity_ids: Set | List = None):
        """
        Applies the WordNet hierarchy to create a directed graph from entity names or IDs.
        
        Args:
            entity_names (set | list): Set or list of entity names (WordNet synsets).
            entity_ids (set | list): Set or list of entity IDs (WordNet offsets).
        Returns:
            G (nx.DiGraph): Directed graph representing the WordNet hierarchy.
        """
        def get_hypernym_paths(synset):
            paths = synset.hypernym_paths()
            path = paths[0] # if multiple paths, take the first one
            path = [(s.name(),s.name().split('.')[0]) for s in path]
            return path

        assert entity_names is not None or entity_ids is not None, "Either entity_names or entity_ids must be provided."
        entities = entity_names if entity_names is not None else entity_ids
        wn_entrypoint = wn.synset if entity_names is not None else (lambda x: wn.synset_from_pos_and_offset('n', x))
        
        # create graph from WordNet hierarchy
        G = nx.DiGraph()
        for entity in entities:
            synset = wn_entrypoint(entity)
            chain = get_hypernym_paths(synset)

            for i in range(len(chain) - 1):
                parent_id, child_id = chain[i][0], chain[i + 1][0]
                parent, child = chain[i][1], chain[i + 1][1]
                if parent == "root":
                    parent = parent + " (entity)"  # to avoid conflicts with the root node
                if child == "root":
                    child = child + " (entity)"
                if not G.has_node(parent_id):
                    G.add_node(parent_id, label=parent)
                if not G.has_node(child_id):
                    G.add_node(child_id, label=child)
                if not G.has_edge(parent_id, child_id):
                    G.add_edge(parent_id, child_id)

        logger.info(f"Created WordNet graph with {G.number_of_nodes()} nodes.")
        return G

    def _map_train_eval_ids(self, G: nx.DiGraph):
        """
        Maps the level ids of the evaluation datasets to the level ids of the training dataset.
        This is necessary because the evaluation datasets may have different level ids than the training dataset.

        Args:
            G (nx.DiGraph): The graph representing the hierarchy of the evaluation dataset.
        Returns:
            G (nx.DiGraph): The graph with updated level ids based on the training dataset.
            number_of_leaves (int): The number of leaf nodes in the graph.
        """
        logger.info("Using train ids to assign level ids for evaluation datasets.")
        dataset_folder = os.path.basename(self.dataset_path)
        train_split = "-".join(self.dataset_name.split("-")[:-1]) + "-train"
        train_graph, _ = load_graph_from_file(os.path.join("metadata", dataset_folder, train_split, "hierarchy.json"))
        train_labels_to_ids = {data['label']:node for node, data in train_graph.nodes(data=True)}
        train_labels_to_ids.pop(self._root_label, None)  # Remove root label if exists
        current_labels_to_ids = {data['label']:node for node, data in G.nodes(data=True)}
        ids_mapping = {current_labels_to_ids[label]: train_labels_to_ids[label] for label in current_labels_to_ids if label in train_labels_to_ids}
        ids_mapping[current_labels_to_ids[self._root_label]] = current_labels_to_ids[self._root_label]  # Ensure root node maps to itself
        G = nx.relabel_nodes(G, ids_mapping, copy=True)
        logger.info(f"Assigned {len(ids_mapping)} level ids based on training dataset.")

        leaves = [n for n in G.nodes if G.out_degree(n) == 0]
        return G, len(leaves)

    def _assign_level_ids(self, G: nx.DiGraph):
        """
        Assigns unique integer ids to nodes in the hierarchy graph based on their hierarchy levels.
        This method ensures that leaf nodes are assigned the lowest ids, and internal nodes are assigned ids based on their depth in the hierarchy.

        Args:
            G (nx.DiGraph): The graph representing the hierarchy.
        Returns:
            G (nx.DiGraph): The graph with updated level ids.
            number_of_leaves (int): The number of leaf nodes in the graph.
        """
        # if "val" in self.dataset_name or "test" in self.dataset_name:
        #     return self._map_train_eval_ids(G)
        
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
    
    def _add_virtual_root(self, G: nx.DiGraph):
        """
        Adds a virtual root node to the graph if there is no root.
        A virtual root node is added to connect all nodes that have no incoming edges.
        This is useful for visualizing the hierarchy as a tree structure.

        Args:
            G (nx.DiGraph): The graph representing the hierarchy.
        Returns:
            G (nx.DiGraph): The graph with a virtual root node added.
        """
        logger.info("Adding virtual root node to the graph.")
        progressive_ids = {n:i for i, n in enumerate(G.nodes)}
        G = nx.relabel_nodes(G, progressive_ids, copy=True)  # Ensure nodes have unique ids
        # Find nodes with no incoming edges
        roots = [n for n in G.nodes if G.in_degree(n) == 0]

        if len(roots) == 1:
            logger.info("Only one root candidate found, no need to add a virtual root.")
            return G

        # Add a virtual root node
        root_id = G.number_of_nodes()
        G.add_node(root_id, label=self._root_label)

        # Connect root to all current root candidates
        for node in roots:
            G.add_edge(root_id, node)
        logger.info(f"Virtual root added with {G.number_of_nodes()} total nodes.")
        return G

    def _tree_depth(self, G: nx.DiGraph, root: int):
        """
        Computes the depth of the tree rooted at the given node.

        Args:
            G (nx.DiGraph): The graph representing the hierarchy.
            root (int): The root node of the subtree.

        Returns:
            int: The depth of the tree.
        """
        lengths = nx.single_source_shortest_path_length(G, root)
        return max(lengths.values())

    def _extract_subtree(self, G: nx.DiGraph, root: int, max_depth: int, subsample: float = 1.0):
        """
        Extracts a subtree from the graph starting from the given root node.
        The subtree is pruned based on the specified maximum depth and subsampling factor.

        Args:
            G (nx.DiGraph): The graph representing the hierarchy.
            root (int): The root node of the subtree.
            max_depth (int): The maximum depth of the subtree to extract.
            subsample (float): Subsampling factor to reduce the size of the subtree. Default is 1.0 (no subsampling).
        Returns:
            G_sub (nx.DiGraph): The extracted subtree graph.
        """
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

    def _visualize_hierarchy(self, G: nx.DiGraph, number_of_nodes: int = None, max_depth: int = 3):
        """
        Visualizes the hierarchy graph using a radial layout.

        Args:
            G (nx.DiGraph): The graph representing the hierarchy.
            number_of_nodes (int): Total number of nodes in the graph. If None, uses the number of nodes in G.
            max_depth (int): Maximum depth of the tree to visualize.
        Returns:
            None: Saves the visualization as a PNG file in the hierarchy_files_path.
        """
        # Compute radial layout
        def radial_layout(G: nx.DiGraph, root: int = 0, depth: int = 3):
            """
            Computes a radial layout for the graph G starting from the root node.
            The nodes are arranged in concentric circles based on their depth in the tree.

            Args:
                G (nx.DiGraph): The graph to layout.
                root (int): The root node of the tree.
                depth (int): The maximum depth of the tree.
            Returns:
                pos (dict): A dictionary mapping nodes to their (x, y) positions in the radial layout.
            """
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

    def _visualize_hierarchy_from_file(self, json_file: str):
        """
        Visualizes the hierarchy from a JSON file.

        Args:
            json_file (str): Path to the JSON file containing the hierarchy data.
        Returns:
            None: Displays the hierarchy as a radial tree using Plotly's Sunburst chart.
        """
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

    ### (End) Helper methods

    def create_hierarchy_file(self):
        """
        Creates a hierarchy file for the specified dataset.
        This method generates a directed graph representing the hierarchy of classes in the dataset,
        assigns unique integer ids to the nodes, and exports the hierarchy in JSON format.
        It also exports dataset details in Markdown format and visualizes the hierarchy.
        The hierarchy is created based on the dataset name and the available methods for different datasets.
        The hierarchy is saved in the specified hierarchy_files_path.

        Args:
            None
        Returns:
            None: The hierarchy file is saved in the specified hierarchy_files_path.
        """
        logger.info(f"Creating hierarchy file for dataset: {self.dataset_name}")
        
        if "ade20k" in self.dataset_name:
            if "wordnet" in self.dataset_name:
                logger.info("Creating hierarchy for ADE20K using WordNet.")
                G = self._ade20k_wordnet()
            elif "coarse_to_fine" in self.dataset_name:
                logger.info("Creating hierarchy for ADE20K using coarse-to-fine.")
                G = self._ade20k_coarse_to_fine()
            elif "scene_cls" in self.dataset_name:
                split = self.dataset_name.split("-")[-1]
                split = "training" if split == "train" else "validation"
                logger.info(f"Creating hierarchy for ADE20K scene classification ({split}).")
                G = self._ade20k_scene_cls(split)
        elif self.dataset_name == "babel":
            raise NotImplementedError("Babel hierarchy creation is not implemented yet.")
            # G = self._babel()
        elif "biotrove" in self.dataset_name:
            if "balanced" in self.dataset_name:
                split = "Balanced"
            elif "lifestages" in self.dataset_name:
                split = "LifeStages"
            elif "unseen" in self.dataset_name:
                split = "Unseen"
            G = self._biotrove(split)
        elif self.dataset_name == "cityscapes":
            G = self._cityscapes()
        elif self.dataset_name == "mapillary":
            G = self._mapillary()
        elif self.dataset_name == "imagenet21k":
            G = self._imagenet21k()
        elif self.dataset_name == "imagenetood":
            G = self._imagenetood()
        elif self.dataset_name == "objects365":
            G = self._objects365()
        elif "paco" in self.dataset_name:
            subdataset, split = self.dataset_name.split("-")[1:]
            G = self._paco(subdataset, split)
        elif self.dataset_name == "rare_species":
            G = self._rare_species()
        elif self.dataset_name == "tree_of_life":
            G = self._tree_of_life()
        elif self.dataset_name == "visual_genome":
            G = self._visual_genome()
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
        from utils.graph_utils import plot_hierarchy
        plot_hierarchy(os.path.join(self.hierarchy_files_path, "hierarchy.json"))

def main():
    parser = argparse.ArgumentParser(description="Create hierarchy file for a dataset.")
    parser.add_argument("dataset_name", type=str, choices=HierarchyFileCreator.datasets, help="Name of the dataset")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset directory")
    parser.add_argument("hierarchy_file_path", type=str, help="Path to save the hierarchy file")

    args = parser.parse_args()

    creator = HierarchyFileCreator(args.dataset_name, args.dataset_path, args.hierarchy_file_path)
    creator.create_hierarchy_file()
    logger.info(f"Hierarchy file created at {args.hierarchy_file_path}")


def scrape_objects365():
    """
    Scrape the Objects365 hierarchy from the website https://www.objects365.org/explore.html.
    """
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from bs4 import BeautifulSoup
    import time

    # Set up Selenium with headless Chrome
    options = Options()
    options.add_argument("--headless")  # Run in headless mode
    service = Service()  # Assumes chromedriver is in your PATH
    driver = webdriver.Chrome(service=service, options=options)

    try:
        # Navigate to the Objects365 Explore page
        url = "https://www.objects365.org/explore.html"
        driver.get(url)

        # Wait for the page to load completely
        time.sleep(5)  # Adjust the sleep time as necessary

        # Locate the element to click (e.g., a button or link)
        # Replace 'element_id' with the actual ID or use other locating strategies
        hierarchy = {}
        for supercat_id in range(11):
            element_to_click = driver.find_element(By.XPATH, f"//button[@onclick='creatCategoryButton({supercat_id})']")
            element_to_click.click()

            # Wait for the new content to load after the click
            time.sleep(5)  # Adjust as necessary based on content load time

            # Get the updated page source and parse with BeautifulSoup
            soup = BeautifulSoup(driver.page_source, 'html.parser')

            supercategory_name = soup.find('button', class_='btn btn-dark btn-sm active').text.strip()
            hierarchy[supercategory_name] = []
            # Now you can use BeautifulSoup to find and extract the desired content
            # For example, extracting all object categories
            categories = soup.find_all('button', class_='btn btn-default btn-block btn-sm')  # Replace with actual class

            for category in categories:
                hierarchy[supercategory_name].append(category.text.strip())

    finally:
        # Close the Selenium driver
        driver.quit()

    with open("../datasets/Objects365/objects365_hierarchy.json", "w") as f:
        json.dump(hierarchy, f, indent=4)


if __name__ == "__main__":
    main()
    # Run the script from command line
    # python create_hierarchy_file.py tree_of_life /path/to/dataset /path/to/hierarchy_file.json
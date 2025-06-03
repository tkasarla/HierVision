import json

import networkx as nx
import numpy as np


def load_graph_from_file(file_path: str) -> nx.DiGraph:
    with open(file_path, "r") as json_file:
        graph_dict = json.load(json_file)
    #search for root node
    root_name =  'root'  
    idx = [root_name == i['label'].lower() for i in graph_dict["nodes"]].index(True)
    root_id = graph_dict["nodes"][idx]["id"]

    graph = nx.node_link_graph(graph_dict)

    return graph,root_id
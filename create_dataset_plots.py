from ast import arg, parse
import json
from count_classes import count_leaves, count_leaves_of_parent
import os
import networkx as nx
from tqdm import tqdm
import plotly.express as px
import kaleido

def get_hierarchy_files(hierarchies_directory="./HierVision/hierarchies/"):
    """
    Get all hierarchy files from the specified directory.
    Returns:
        list: A list of paths to hierarchy files.
    """
    # find all the .json files in all the sub and sub subdirectories of the hierarchies directory
    hierarchy_files = []
    for root, dirs, files in os.walk(hierarchies_directory):
        for file in files:
            if file.endswith(".json") and not file.startswith("nested") and not file.startswith("imagenet100"):
                hierarchy_file = os.path.join(root, file)
                # print(f"Found hierarchy file: {hierarchy_file}")
                hierarchy_files.append(hierarchy_file)
    # remove filepaths with these names in them
    for filepath in hierarchy_files:
        if "heirarchy_sun360_2012_with_count" in filepath or "gene_ontology_mapping.json" in filepath or "ADE20K" in filepath or "bbox_labels_600_hierarchy" in filepath or "imagenet100" in filepath:
            print(f"Removing unformatted hierarchy file: {filepath}")
            hierarchy_files.remove(filepath)
    print(f"Found {len(hierarchy_files)} hierarchy files in the directory: {hierarchies_directory}")
    # print(f"files are {hierarchy_files}")
    return hierarchy_files

# define a hierarchy class below
class Hierarchy:
    def __init__(self, hierarchy_file):
        self.hierarchy_file = hierarchy_file
        self.graph = self.load_hierarchy_file(hierarchy_file)

    def load_hierarchy_file(self, json_file):
        """
        Load a hierarchy file and return the graph object.
        Args:
            json_file (str): Path to the hierarchy JSON file.
        Returns:
            nx.DiGraph: A directed graph representing the hierarchy.
        """
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Create a directed graph
        G = nx.DiGraph()

        # Add nodes
        for node in data["nodes"]:
            G.add_node(node["id"])

        # Add edges
        for link in data["links"]:
            G.add_edge(link["source"], link["target"])

        return G

    def get_number_of_nodes(self):
        """
        Get the number of nodes in the hierarchy.
        Returns:
            int: Number of nodes in the hierarchy.
        """
        return self.graph.number_of_nodes()
    
    def get_depth(self):
        """
        Get the depth of the hierarchy.
        Returns:
            int: Depth of the hierarchy.
        """
        return nx.dag_longest_path_length(self.graph)
    
    def get_number_of_leaves(self):
        """
        Get the number of leaf nodes in the hierarchy.
        Returns:
            int: Number of leaf nodes in the hierarchy.
        """
        return len([n for n in self.graph.nodes if self.graph.out_degree(n) == 0])
    
    def get_average_branching_factor(self):
        """
        Get the average branching factor of the hierarchy.
        Returns:
            float: Average branching factor of the hierarchy.
        Explanation: 
        The average branching factor is calculated as the total number of branches (outgoing edges) divided by the number of nodes.
        """
        total_branches = sum(self.graph.out_degree(n) for n in self.graph.nodes)
        return total_branches / self.get_number_of_nodes() if self.get_number_of_nodes() > 0 else 0


def process_hierarchy_files(hierarchy_files, show_print=True):
    """
    Process all hierarchy files to count leaves and optionally count leaves of a specific parent.
    """
    dataset_info = {}
    for hierarchy_file in tqdm(hierarchy_files):
        print(f"Processing hierarchy file: {hierarchy_file}")
        hierarchy_obj = Hierarchy(hierarchy_file)
        dataset_name = hierarchy_file.split("/")[3]
        dataset_info[dataset_name] = {
            "number_of_nodes": hierarchy_obj.get_number_of_nodes(),
            "depth": hierarchy_obj.get_depth(),
            "number_of_leaves": hierarchy_obj.get_number_of_leaves(),
            "average_branching_factor": hierarchy_obj.get_average_branching_factor()
        }
        if show_print:
            print(f"Dataset Info for {dataset_name}:")
            print(f"Number of Nodes: {dataset_info[dataset_name]['number_of_nodes']}")
            print(f"Depth: {dataset_info[dataset_name]['depth']}")
            print(f"Number of Leaves: {dataset_info[dataset_name]['number_of_leaves']}")
            print(f"Average Branching Factor: {dataset_info[dataset_name]['average_branching_factor']:.2f}")
            print("-" * 40)
    # save dataset info to a json file
    with open("dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=4)
    # save to csv   
    import pandas as pd
    df = pd.DataFrame(dataset_info).T
    df.to_csv("dataset_info.csv", index_label="dataset_name")
    return dataset_info

def plot_histogram_of_node_counts(dataset_info, bins=30, binsize=100):
    """
- **Histogram of Node Counts per Hierarchy**
    - **X-axis**: Number of nodes (bucketed)
    - **Y-axis**: Frequency (# of datasets)
    - Shows diversity from shallow to deep hierarchies (e.g., CIFAR-100 vs TreeOfLife-10M)
    """
    if type(dataset_info) == str:
        print(f"dataset info is a string: {dataset_info}")
        with open(dataset_info, 'r') as f:
            dataset_info = json.load(f)
        print(f"Plotting histogram of node counts for {len(dataset_info)} datasets.")
    if type(dataset_info) == dict:
        print(f"Plotting histogram of node counts for {len(dataset_info)} datasets.")
    else:
        raise ValueError("No dataset info provided. Please process hierarchy files first.")

    node_counts = [info['number_of_nodes'] for info in dataset_info.values()]
    for dataset_name, info in dataset_info.items():
        print(f"{dataset_name}: {info['number_of_nodes']} nodes")
    # plot based on bin size
    # bin the node counts based on the binsize
    node_counts_binned = [count // binsize * binsize for count in node_counts]
    print(f"Node counts binned: {node_counts_binned}")
    fig = px.histogram(node_counts_binned, nbins=bins, title="Histogram of Node Counts per Hierarchy")
    fig.update_layout(
        xaxis_title="Number of Nodes",
        yaxis_title="Frequency (# of Datasets)",
        bargap=0.2
    )
    fig.show()

    ## uncomment to plot based on bins
    # fig = px.histogram(node_counts, nbins=bins, title="Histogram of Node Counts per Hierarchy")
    # fig.update_layout(
    #     xaxis_title="Number of Nodes",
    #     yaxis_title="Frequency (# of Datasets)",
    #     bargap=0.2
    # )
    fig.show() 
    fig.write_image("histogram_of_node_counts.png")

def plot_semantic_granularity_histogram(dataset_info, bins=30, color="blue"):
    """
    - **Histogram of Maximum Tree Depths**
    - Useful to show semantic granularity (e.g., 2–3 levels = coarse; 7+ = fine-grained)
    """
    if type(dataset_info) == str:
        print(f"dataset info is a string: {dataset_info}")
        with open(dataset_info, 'r') as f:
            dataset_info = json.load(f)
        print(f"Plotting histogram of maximum tree depths for {len(dataset_info)} datasets.")
    if type(dataset_info) == dict:
        print(f"Plotting histogram of maximum tree depths for {len(dataset_info)} datasets.")
    else:
        raise ValueError("No dataset info provided. Please process hierarchy files first.")

    max_depths = [info['depth'] for info in dataset_info.values()]
    fig = px.histogram(max_depths, nbins=bins, title="Histogram of Maximum Tree Depths")
    fig.update_layout(
        xaxis_title="Maximum Tree Depth",
        yaxis_title="Frequency (# of Datasets)",
        bargap=0.2,
        template="plotly_white"
    )
    fig.update_traces(marker_color=color)
    fig.show()
    fig.write_image("histogram_of_maximum_tree_depths.png")

def plot_max_depth_by_nodes_scatterplot(dataset_info):
    """
    - **Scatter Plot of Maximum Depth vs Node Count**
    - Useful to show relationship between hierarchy depth and number of nodes
    """
    if type(dataset_info) == str:
        print(f"dataset info is a string: {dataset_info}")
        with open(dataset_info, 'r') as f:
            dataset_info = json.load(f)
        print(f"Plotting scatter plot of maximum depth vs node count for {len(dataset_info)} datasets.")
    if type(dataset_info) == dict:
        print(f"Plotting scatter plot of maximum depth vs node count for {len(dataset_info)} datasets.")
    else:
        raise ValueError("No dataset info provided. Please process hierarchy files first.")

    # Extract dataset names, maximum depths, and node counts
    dataset_names = list(dataset_info.keys())
    max_depths = [info['depth'] for info in dataset_info.values()]
    node_counts = [info['number_of_nodes'] for info in dataset_info.values()]

    # Conditionally add text for datapoints with >1000 nodes
    text_labels = [
        name if nodes > 1000 else None
        for name, nodes in zip(dataset_names, node_counts)
    ]


    # Create scatter plot with dataset names as hover text
    fig = px.scatter(
        x=node_counts,
        y=max_depths,
        title="Maximum Depth vs Node Count",
        hover_name=dataset_names,  # Add dataset names as hover text only
        opacity=0.5,  # Set transparency for overlapping points
        text=text_labels,  # Add text conditionally
        # color=node_counts,  # Optional: color by node count
        # set a static color for all points
        color_discrete_sequence=["green"]  # Set a static color for all points
    )

    # Update marker size and layout for log scale
    fig.update_traces(
        marker=dict(size=20),  # Increase marker size
        textposition="top center"  # Position text above datapoints
    )
    fig.update_layout(
        xaxis_title="Number of Nodes (Log Scale)",
        yaxis_title="Maximum Tree Depth",
        xaxis_type="log",  # Apply log scale to x-axis
        template="plotly_white"
    )

    # Show the plot and save it as an image
    fig.show()
    fig.write_image("max_depth_vs_node_count_scatterplot_log_scale.png")


# accept command line arguments as input
import argparse
parser = argparse.ArgumentParser(description="Process hierarchy files and plot histograms.")
parser.add_argument("--hierarchies_directory", type=str, default="./HierVision/hierarchies/", help="Directory containing hierarchy files.")
parser.add_argument("--show_print", action='store_true', help="Whether to print dataset info during processing.")
parser.add_argument("--dataset_info_filepath", type=str, default="dataset_info.json", help="Filepath to save dataset info JSON.")
parser.add_argument("--bins", type=int, default=30, help="Number of bins for the histogram of node counts.")
parser.add_argument("--binsize", type=int, default=100, help="Size of each bin for the histogram of node counts.")
parser.add_argument("--plot_semantic_granularity", action='store_true', help="Whether to plot the semantic granularity histogram.")
parser.add_argument("--plot_node_counts", action='store_true', help="Whether to plot the histogram of node counts.")
parser.add_argument("--color", type=str, default="pink", help="Color for the semantic granularity histogram.")

args = parser.parse_args()

if __name__ == "__main__":

    # Example usage:
    # python create_dataset_plots.py --dataset_info_filepath dataset_info.json --bins 100

    if args.dataset_info_filepath != None and os.path.exists(args.dataset_info_filepath):
        dataset_info = args.dataset_info_filepath
        print(f"Skipping processing hierarchies, using existing hierarchy info from {args.dataset_info_filepath}")
        plot_max_depth_by_nodes_scatterplot(dataset_info)
        if args.plot_semantic_granularity:
            print("Plotting semantic granularity histogram...")
            plot_semantic_granularity_histogram(dataset_info, bins=args.bins)
        if args.plot_node_counts:
            print("Plotting histogram of node counts...")
            plot_histogram_of_node_counts(bins=args.bins, dataset_info="dataset_info.json", binsize=args.binsize)
    else:
        print(f"Processing hierarchy files from directory: {args.hierarchies_directory}")
        hierarchy_files = get_hierarchy_files(args.hierarchies_directory)
        dataset_info = process_hierarchy_files(hierarchy_files, show_print=args.show_print)
        plot_histogram_of_node_counts(bins=args.bins, dataset_info=dataset_info)
        print("Processing complete. Dataset info saved to 'dataset_info.json'.")

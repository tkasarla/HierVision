from ast import arg, parse
import json
from platform import node
from turtle import color
from count_classes import count_leaves, count_leaves_of_parent
import os
import networkx as nx
from tqdm import tqdm
import plotly.express as px
import kaleido
import numpy as np

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
                print(f"Found hierarchy file: {hierarchy_file}")
                hierarchy_files.append(hierarchy_file)
    # remove filepaths with these names in them
    for filepath in hierarchy_files:
        if "heirarchy_sun360_2012_with_count" in filepath or "gene_ontology_mapping.json" in filepath or "bbox_labels_600_hierarchy" in filepath or "imagenet100" in filepath:
            print(f"Removing unformatted hierarchy file: {filepath}")
            hierarchy_files.remove(filepath)
    print(f"Found {len(hierarchy_files)} hierarchy files in the directory: {hierarchies_directory}")
    print(f"files are {hierarchy_files}")
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
        The average branching factor is calculated as the average number of children per internal node (meaning excluding leaf nodes).
        """
        if self.graph.number_of_nodes() == 0:
            return 0.0
        total_children = sum(self.graph.out_degree(n) for n in self.graph.nodes if self.graph.out_degree(n) > 0)
        internal_nodes_count = sum(1 for n in self.graph.nodes if self.graph.out_degree(n) > 0)
        if internal_nodes_count == 0:
            return 0.0
        return total_children / internal_nodes_count


def process_hierarchy_files(hierarchy_files, show_print=True):
    """
    Process all hierarchy files to count leaves and optionally count leaves of a specific parent.
    """
    dataset_info = {}
    for hierarchy_file in tqdm(hierarchy_files):
        print(f"Processing hierarchy file: {hierarchy_file}")
        hierarchy_obj = Hierarchy(hierarchy_file)
        dataset_name = hierarchy_file.split("/")[3]
        if hierarchy_file.split("/")[3] == "medical_imaging_datasets":
            print(f"Found medical_imaging_datasets hierarchy file: {hierarchy_file}")
            dataset_name = hierarchy_file.split("/")[4]
            print(f"dataset name here is {dataset_name}")
        if "_" in dataset_name:
            # replace the _ by space
            dataset_name = dataset_name.replace("_", " ")
            # capitalize the first letter of each word
            dataset_name = dataset_name.title()
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

def increase_font_size(fig):
    # Update layout to increase font sizes
    fig.update_layout(
        title=dict(
            font=dict(size=30)  # Increase title font size
        ),
        xaxis=dict(
            title=dict(
                font=dict(size=24)  # Increase x-axis title font size
            )
        ),
        yaxis=dict(
            title=dict(
                font=dict(size=24)  # Increase y-axis title font size
            )
        ),
        font=dict(size=20)  # Increase font size for text labels
    )

def plot_histogram_of_node_counts(dataset_info, binsize=50):
    """
    Plot a histogram of node counts per hierarchy using plotly.express.histogram.
    Args:
        dataset_info (dict or str): Dataset information or path to JSON file.
        binsize (int): Size of each bin (e.g., 50 for bins like 0-50, 50-100, etc.).
    """
    if type(dataset_info) == str:
        print(f"dataset info is a string: {dataset_info}")
        with open(dataset_info, 'r') as f:
            dataset_info = json.load(f)
        print(f"Plotting histogram of node counts for {len(dataset_info)} datasets.")
    elif type(dataset_info) == dict:
        print(f"Plotting histogram of node counts for {len(dataset_info)} datasets.")
    else:
        raise ValueError("No dataset info provided. Please process hierarchy files first.")

    # Extract node counts
    node_counts = [info['number_of_nodes'] for info in dataset_info.values()]
    print(f"Node counts: {node_counts}")

    # Create histogram using plotly.express
    fig = px.histogram(
        x=node_counts,
        nbins=int(max(node_counts) / binsize),  # Define number of bins based on binsize
        labels={"x": "Number of Nodes", "y": "Frequency (# of Datasets)"},
        title="Histogram of Node Counts",
        color_discrete_sequence=["blue"]  # Set a static color for the bars
    )

    # Update layout for better visualization
    fig.update_layout(
        xaxis=dict(title="Number of Nodes"),
        yaxis=dict(title="Frequency (# of Datasets)"),
        bargap=0.2,  # Adjust spacing between bars
        template="plotly_white"
    )

    increase_font_size(fig)  # Increase font size for better readability
    fig.show()


def plot_histogram_of_node_counts2(dataset_info, bins=30, binsize=100):
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
    print(dataset_info)
    node_counts = [info['number_of_nodes'] for info in dataset_info.values()]
    print(node_counts)
    log_bins = np.logspace(np.log10(min(node_counts)), np.log10(max(node_counts)), num=20)

    # Calculate histogram frequencies
    hist, bin_edges = np.histogram(node_counts, bins=log_bins)

    print(f"node counts: {node_counts}")
    for dataset_name, info in dataset_info.items():
        print(f"{dataset_name}: {info['number_of_nodes']} nodes")

    bar_width = 10  # Set a fixed width for the bars
    import plotly.graph_objects as go
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bin_edges[:-1],  # Bin start points
            y=hist,  # Frequencies
            width=[bar_width] * len(hist),  # Fixed width for all bars
            name="Node Counts",
        )
    )

    fig.update_layout(
        xaxis=dict(
            title="Number of Nodes (Log Scale)",
            type="log",  # Apply log scale to x-axis
        ),
        yaxis=dict(
            title="Frequency (# of Datasets)",
        ),
        title="Histogram of Node Counts",
        bargap=0.2,  # Adjust spacing between bars
    )

    increase_font_size(fig)  # Increase font size for better readability
    fig.show()

def plot_semantic_granularity_histogram(dataset_info, color="blue", bins=None):
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
    # use custom bins
    if bins:
        bins = np.arange(0, max(max_depths) + 1, bins)  # Create bins from 0 to max depth with specified size
    fig = px.histogram(
        max_depths, 
        title="Histogram of Maximum Tree Depths",
        color_discrete_sequence=["blue"]  # Set a static color for the histogram
        )
    
    fig.update_layout(
        xaxis_title="Maximum Tree Depth",
        yaxis_title="Frequency (# of Datasets)",
        bargap=0.2,
        template="plotly_white",
        # xaxis_type="log",  # Apply log scale to x-axis
        showlegend=False  # Remove the legend
    )
    increase_font_size(fig)  # Increase font size for better readability
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
    branching_factors = [info['average_branching_factor'] for info in dataset_info.values()]
    branching_factors_weighted = 12+ np.array(branching_factors)  # Scale the branching factors for better visibility in the plot
    # fixed size each circle the same
    branching_factors_fixed = [12] * len(branching_factors_weighted)  # Set a fixed size for all circles
    # Conditionally add text for datapoints with >1000 nodes
    text_labels = [
        name if nodes > 1000 else None
        for name, nodes in zip(dataset_names, node_counts)
    ]


    # Create scatter plot with dataset names as hover text
    fig = px.scatter(
        x=node_counts,
        y=max_depths,
        size=branching_factors_fixed,  # Circle size proportional to average branching factor
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
        # marker=dict(size=20),  # Increase marker size
        textposition="top center"  # Position text above datapoints
    )
    fig.update_layout(
        xaxis_title="Number of Nodes (Log Scale)",
        yaxis_title="Maximum Tree Depth",
        xaxis_type="log",  # Apply log scale to x-axis
        template="plotly_white"
    )
    increase_font_size(fig)  # Increase font size for better readability

    # Show the plot and save it as an image
    fig.show()
    fig.write_image("max_depth_vs_node_count_scatterplot_log_scale.png")

def plot_average_branching_factor(dataset_info):
    """
    Plot the average branching factor per dataset.
    X-axis: Dataset name
    Y-axis: Average branching factor
    """
    # Extract dataset names and average branching factors
    if type(dataset_info) == str:
        print(f"dataset info is a string: {dataset_info}")
        with open(dataset_info, 'r') as f:
            dataset_info = json.load(f)
    dataset_names = list(dataset_info.keys())
    branching_factors = [info['average_branching_factor'] for info in dataset_info.values()]
    # make them sorted
    sorted_indices = np.argsort(branching_factors)
    dataset_names = [dataset_names[i] for i in sorted_indices]
    branching_factors = [branching_factors[i] for i in sorted_indices]

    # Create bar plot
    fig = px.bar(
        x=dataset_names,
        y=branching_factors,
        labels={"x": "Dataset Name", "y": "Average Branching Factor"},
        title="Average Branching Factor per Dataset",
        color_discrete_sequence=["purple"]  # Set a static color for the bars
    )

    # Update layout for better visualization
    fig.update_layout(
        xaxis=dict(tickangle=-45),  # Rotate x-axis labels for readability
        template="plotly_white"
    )

    # Increase font size for better readability
    increase_font_size(fig)

    # Show the plot
    fig.show()
    fig.write_image("average_branching_factor_per_dataset.png")

def plot_radar_min_max(dataset_info):
    """
    Create a radar plot for the minimum and maximum values of:
    - Number of nodes
    - Depth
    - Number of leaves
    - Average branching factor
    """
    if type(dataset_info) == str:
        print(f"dataset info is a string: {dataset_info}")
        with open(dataset_info, 'r') as f:
            dataset_info = json.load(f)
    # Extract values
    node_counts = [info['number_of_nodes'] for info in dataset_info.values()]
    depths = [info['depth'] for info in dataset_info.values()]
    leaf_counts = [info['number_of_leaves'] for info in dataset_info.values()]
    branching_factors = [info['average_branching_factor'] for info in dataset_info.values()]

    # Calculate min and max for each metric
    metrics = {
        "Number of Nodes": (min(node_counts), max(node_counts)),
        "Depth": (min(depths), max(depths)),
        "Number of Leaves": (min(leaf_counts), max(leaf_counts)),
        "Average Branching Factor": (min(branching_factors), max(branching_factors))
    }

    # Prepare data for radar plot
    categories = list(metrics.keys())
    min_values = [np.log10(metrics[metric][0]) if metrics[metric][0] > 0 else 0 for metric in categories]
    max_values = [np.log10(metrics[metric][1]) if metrics[metric][1] > 0 else 0 for metric in categories]
    print(f"Categories: {categories}")
    print(f"Log Min values: {min_values}")
    print(f"Log Max values: {max_values}")

    # Create radar plot
    import plotly.graph_objects as go
    fig = go.Figure()

    # Add min values
    fig.add_trace(go.Scatterpolar(
        r=min_values,
        theta=categories,
        fill='toself',
        name='Minimum Values (Log Scale)'
    ))

    # Add max values
    fig.add_trace(go.Scatterpolar(
        r=max_values,
        theta=categories,
        fill='toself',
        name='Maximum Values (Log Scale)'
    ))

    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                title="Log Scale"
            )
        ),
        title="Radar Plot of Min and Max Values Across Metrics (Log Scale)",
        template="plotly_white"
    )

    # Show the plot
    fig.show()
    fig.write_image("radar_plot_min_max_log_scale.png")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_hexbin_with_marginals(dataset_info):
    """
    Hexbin plot of Max Depth vs. Average Branching Factor with marginal distributions.
    """
    if type(dataset_info) == str:
        with open(dataset_info, 'r') as f:
            dataset_info = json.load(f)
    elif type(dataset_info) == dict:
        print(f"Plotting hexbin with marginals for {len(dataset_info)} datasets.")
    else:
        raise ValueError("No dataset info provided. Please process hierarchy files first.")
    # Extract data
    max_depths = [info['depth'] for info in dataset_info.values()]
    branching_factors = [info['average_branching_factor'] for info in dataset_info.values()]

    # Create figure and gridspec for marginal distributions
    fig = plt.figure(figsize=(10, 8))
    grid = plt.GridSpec(4, 4, hspace=0.5, wspace=0.5)

    # Main hexbin plot
    ax_main = fig.add_subplot(grid[1:4, 0:3])
    hb = ax_main.hexbin(branching_factors, max_depths, gridsize=15, cmap='cool', mincnt=1)
    ax_main.set_xlabel("Average Branching Factor")
    ax_main.set_ylabel("Maximum Depth")
    ax_main.set_title("Hexbin Plot of Max Depth vs. Branching Factor")
    cb = fig.colorbar(hb, ax=ax_main)
    cb.set_label("Frequency")

    # Marginal distribution for branching factors
    ax_top = fig.add_subplot(grid[0, 0:3], sharex=ax_main)
    sns.histplot(branching_factors, bins=30, kde=True, ax=ax_top, color="blue")
    ax_top.set_ylabel("Frequency")
    ax_top.set_title("Marginal Distribution of Branching Factor")

    # Marginal distribution for max depths
    ax_right = fig.add_subplot(grid[1:4, 3], sharey=ax_main)
    sns.histplot(max_depths, bins=30, kde=True, ax=ax_right, color="blue", orientation="horizontal")
    ax_right.set_xlabel("Frequency")
    ax_right.set_title("Marginal Distribution of Max Depth")

    # Adjust layout
    plt.tight_layout()
    plt.show()
    fig.savefig("hexbin_with_marginals.png")

import plotly.express as px

def plot_treemap(dataset_info, color_metric="depth"):
    """
    Create a treemap where:
    - Block size: Total node count
    - Block color: Max depth or average branching factor
    - Block label: Dataset name
    Args:
        dataset_info (dict): Dictionary containing dataset information.
        color_metric (str): Metric to use for block color ("depth" or "average_branching_factor").
    """
    if type(dataset_info) == str:
        print(f"dataset info is a string: {dataset_info}")
        with open(dataset_info, 'r') as f:
            dataset_info = json.load(f)
    elif type(dataset_info) != dict:
        raise ValueError("No valid dataset info provided. Please process hierarchy files first.")

    # Prepare data for treemap
    dataset_names = list(dataset_info.keys())
    node_counts = [info['number_of_nodes'] for info in dataset_info.values()]
    color_values = [info[color_metric] for info in dataset_info.values()]
    normalized_color_values = np.array(color_values) / np.max(color_values)  # Normalize color values for better visualization
    print(f"Dataset Names: {dataset_names}")
    print(f"Node Counts: {node_counts}")
    # print(f"Color Values ({color_metric}): {color_values}")
    
    # Filter invalid data
    filtered_data = [
        (name, nodes, color)
        for name, nodes, color in zip(dataset_names, node_counts, normalized_color_values)
        if nodes > 0 and color is not None
    ]

    if not filtered_data:
        raise ValueError("No valid data to plot the treemap.")

    dataset_names, node_counts, normalized_color_values = zip(*filtered_data)

    print(f"Filtered Dataset Names: {dataset_names}")
    print(f"Filtered Node Counts: {node_counts}")
    print(f"Filtered Normalized Color Values: {normalized_color_values}")

    # Create treemap
    fig = px.treemap(
        names=dataset_names,  # Dataset names as labels
        values=node_counts,  # Block size based on node count
        color=normalized_color_values,  # Block color based on the chosen metric
        color_continuous_scale="Viridis",  # Color scale
        title=f"Treemap of Node Count (Block Size) vs {color_metric.capitalize()} (Block Color)"
    )

    # Update layout for better visualization
    fig.update_layout(
        template="plotly_white",
        coloraxis_colorbar=dict(title=color_metric.capitalize())
    )

    # Show the plot
    fig.show()
    fig.write_image(f"treemap_node_count_vs_{color_metric}.png")

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
parser.add_argument("--plot_max_depth_by_nodes", action='store_true', help="Whether to plot the scatter plot of maximum depth vs node count.")
parser.add_argument("--plot_average_branching_factor", action='store_true', help="Whether to plot the average branching factor per dataset.")
parser.add_argument("--plot_radar_min_max", action='store_true', help="Whether to plot the radar plot of min and max values across metrics.")
args = parser.parse_args()

if __name__ == "__main__":

    # Example usage:
    # python create_dataset_plots.py --dataset_info_filepath dataset_info.json --bins 100

    if args.dataset_info_filepath != None and os.path.exists(args.dataset_info_filepath):
        dataset_info = args.dataset_info_filepath
        print(f"Skipping processing hierarchies, using existing hierarchy info from {args.dataset_info_filepath}")
        # plot_treemap(dataset_info, color_metric="depth")
        plot_hexbin_with_marginals(dataset_info)
        if args.plot_max_depth_by_nodes:
            plot_max_depth_by_nodes_scatterplot(dataset_info)
        if args.plot_average_branching_factor:
            plot_average_branching_factor(dataset_info)
        if args.plot_radar_min_max:
            plot_radar_min_max(dataset_info)
        if args.plot_semantic_granularity:
            print("Plotting semantic granularity histogram...")
            plot_semantic_granularity_histogram(dataset_info, bins=args.bins)
        if args.plot_node_counts:
            print("Plotting histogram of node counts...")
            plot_histogram_of_node_counts(dataset_info="dataset_info.json", binsize=args.binsize)
    else:
        print(f"Processing hierarchy files from directory: {args.hierarchies_directory}")
        hierarchy_files = get_hierarchy_files(args.hierarchies_directory)
        dataset_info = process_hierarchy_files(hierarchy_files, show_print=args.show_print)
        print("Processing complete. Dataset info saved to 'dataset_info.json'.")

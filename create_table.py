import os
import pandas as pd
import networkx as nx
from typing import List, Tuple
from glob import glob
from utils.graph_utils import load_graph_from_file


def lookup_tasks(file_path: str) -> Tuple[str, str]:
    tasks_db = {

    }

    dataset_key = os.path.basename(os.path.dirname(file_path))

    return tasks_db.get(dataset_key, "vision"), dataset_key


def get_dataset_stats(file_path) -> pd.DataFrame:
    G, root_id = load_graph_from_file(file_path)

    # statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_classes = len([n for n in G.nodes() if G.out_degree(n) == 0])
    max_depth = max(nx.single_source_shortest_path_length(G, root_id).values())
    tasks_for_dataset, dataset_name = lookup_tasks(file_path)

    # create a DataFrame with the statistics
    stats = {
        "Dataset": dataset_name,
        "Nodes": num_nodes,
        "Edges": num_edges,
        "Depth": num_classes,
        "Classes": max_depth,
        "Tasks": tasks_for_dataset
    }

    df = pd.DataFrame([stats])

    return df


def get_final_table(datasets:List[pd.DataFrame], save_path:str=None) -> str:

    table_df = pd.concat(datasets, ignore_index=False)
    table_df.set_index(["Tasks", "Dataset"], inplace=True)
    breakpoint()

    # export to latex booktabs
    # the table should be like the following:
    #
    # \textbf{Dataset} | \textbf{Nodes} | \textbf{Edges} | \textbf{Depth} | \textbf{Classes}
    # \multicolumn{\textbf{Tasks values 1}} # should span over all the coulmns and contain the tasks values
    # dataset 1 | 100 | 200 | 10 | 5 # examples
    # dataset 2 | 150 | 300 | 15 | 7
    # dataset 3 | 200 | 400 | 20 | 10
    # \multicolu\mn{\textbf{Tasks values 2}} # should span over all the coulmns and contain the tasks values
    # dataset 4 | 250 | 500 | 25 | 12
    # dataset 5 | 300 | 600 | 30 | 15

    # Wrap column headers in \textbf{}
    headers = ["\\textbf{Nodes}", "\\textbf{Edges}", "\\textbf{Depth}", "\\textbf{Classes}"]

    table_latex = table_df.to_latex(
        column_format="lcccc",
        header=headers,
        index=True,
        multirow=True,
        multicolumn=True,
        escape=False,  # to allow for special characters
        bold_rows=False,
        longtable=True,
        caption="Dataset Statistics",
        label="tab:dataset_stats"
    )

    print(table_latex)


if __name__ == "__main__":

    path_to_hierarchies = "code/HierVision/hierarchies/**/*.json"
    hierarchy_files = glob(path_to_hierarchies, recursive=True)
    datasets = []
    for file_path in hierarchy_files:
        try:
            stats_df = get_dataset_stats(file_path)
            datasets.append(stats_df)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if datasets:
        get_final_table(datasets, save_path="code/HierVision/hierarchy_stats.tex")
    else:
        print("No datasets found or all datasets failed to process.")
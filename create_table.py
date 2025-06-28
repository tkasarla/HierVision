import os
import pandas as pd
import networkx as nx
from typing import List, Tuple
from glob import glob
from utils.graph_utils import load_graph_from_file
from loguru import logger
import random

dataset_names = {
    "activitynet": "ActivityNet",
    "ADE20K": "ADE20K",
    "aircraft": "FGVC-Aircraft",
    "AWA-2": "AwA2",
    "Bamboo": "Bamboo",
    "BioTrove": "BioTrove",
    "Caltech101": "Caltech-101",
    "cifar100": "CIFAR-100",
    "Cityscapes": "Cityscapes",
    "CMU_mocap": "CMU MoCap",
    "COCO10k": "COCO-10K",
    "Cod10k": "COD10K",
    "Core50": "CORe50",
    "cub": "CUB-200-2011",
    "DLD3V-10K": "DLD3V-10K",
    "EgoObjects": "EgoObjects",
    "fashionpedia": "Fashionpedia",
    "FineSports": "FineSports",
    "GroceryDataset": "Grocery",
    "HDM05": "HDM05",
    "HowTo100M": "HowTo100M",
    "HumanAct12": "HumanAct12",
    "imagenet100": "ImageNet-100",
    "Imagenet21k": "ImageNet-21K",
    "imagenetood": "ImageNet-OOD",
    "IP102": "IP102",
    "Madverse": "MadVerse",
    "MammalNet": "MammalNet",
    "Mapillary_Vistas": "Mapillary Vistas",
    "marine-tree": "Marine Tree",
    "Matador": "Matador",
    "CheXpert": "CheXpert/MIMIC-CXR",
    "DeepLesion": "DeepLesion",
    "MIMIC-CXR": "MIMIC-CXR",
    "OpenCell": "OpenCell",
    "MillionAid": "Million-AID",
    "mini_kinetics": "Mini-Kinetics-200",
    "Mintrec": "MintRec",
    "moments_in_time": "Moments in Time",
    "NABirds": "NABirds",
    "Objects365": "Objects365",
    "openimages": "Open Images",
    "OpenLoris": "OpenLORIS",
    "PACO": "PACO",
    "Pascal": "PASCAL VOC",
    "Portrait-Mode-400": "Portrait Mode 400",
    "PseudoAdverbs": "Pseudo-Adverbs",
    "rare_species": "Rare Species",
    "SomethingSomethingV2": "Something-Something V2",
    "StanfordOnlineProducts": "Stanford Online Products",
    "stanford_cars": "Stanford Cars",
    "SUN360": "SUN360",
    "SUN397": "SUN397",
    "SUN908": "SUN908",
    "TreeOfLife": "Tree of Life",
    "UCF101": "UCF101",
    "vegfru": "VegFru",
    "Visual_Genome": "Visual Genome"
}

dataset_tasks = {
    "activitynet": "Actions / Video",
    "ADE20K": "Scenes / Places",
    "aircraft": "Objects / General",
    "AWA-2": "Biological",
    "Bamboo": "Objects / General",
    "BioTrove": "Biological",
    "Caltech101": "Objects / General",
    "cifar100": "Objects / General",
    "Cityscapes": "Scenes / Places",
    "CMU_mocap": "Objects / General",
    "COCO10k": "Objects / General",
    "Cod10k": "Objects / General",
    "Core50": "Objects / General",
    "cub": "Biological",
    "DLD3V-10K": "Actions / Video",
    "EgoObjects": "Objects / General",
    "fashionpedia": "Objects / General",
    "FineSports": "Actions / Video",
    "GroceryDataset": "Scenes / Places",
    "HDM05": "Actions / Video",
    "HowTo100M": "Actions / Video",
    "HumanAct12": "Actions / Video",
    "imagenet100": "Objects / General",
    "Imagenet21k": "Objects / General",
    "imagenetood": "Objects / General",
    "IP102": "Objects / General",
    "Madverse": "Actions / Video",
    "MammalNet": "Biological",
    "Mapillary_Vistas": "Scenes / Places",
    "marine-tree": "Biological",
    "Matador": "Actions / Video",
    "CheXpert": "Medical",
    "DeepLesion": "Medical",
    "MIMIC-CXR": "Medical",
    "OpenCell": "Medical",
    "MillionAid": "Scenes / Places",
    "mini_kinetics": "Actions / Video",
    "Mintrec": "Actions / Video",
    "moments_in_time": "Actions / Video",
    "NABirds": "Biological",
    "Objects365": "Objects / General",
    "openimages": "Objects / General",
    "OpenLoris": "Objects / General",
    "PACO": "Scenes / Places",
    "Pascal": "Objects / General",
    "Portrait-Mode-400": "Objects / General",
    "PseudoAdverbs": "Actions / Video",
    "rare_species": "Biological",
    "SomethingSomethingV2": "Actions / Video",
    "StanfordOnlineProducts": "Objects / General",
    "stanford_cars": "Objects / General",
    "SUN360": "Scenes / Places",
    "SUN397": "Scenes / Places",
    "SUN908": "Scenes / Places",
    "TreeOfLife": "Biological",
    "UCF101": "Actions / Video",
    "vegfru": "Biological",
    "Visual_Genome": "Scenes / Places"
}

hierarchy_sources = {
    "COCO10k": "\\cite{atigh2022hyperbolic}",
    "Pascal": "\\cite{atigh2022hyperbolic}",
    "ADE20K": "\\cite{atigh2022hyperbolic}",
    "SomethingSomethingV2": "\\cite{mahdisoltani2018effectiveness}",
}

original_format = {
    "activitynet": "JSON",
    "ADE20K": "JSON",
    "aircraft": "TXT",
    "AWA-2": "TXT",
    "Bamboo": "JSON",
    "BioTrove": "CSV",
    "Caltech101": "FOLDER",
    "cifar100": "PKL",
    "Cityscapes": "WEB",
    "CMU_mocap": "WEB",
    "COCO10k": "JSON",
    "Cod10k": "JSON",
    "Core50": "TXT",
    "cub": "JSON",
    "DLD3V-10K": "JSON",
    "EgoObjects": "JSON",
    "fashionpedia": "JSON",
    "FineSports": "PKL",
    "GroceryDataset": "CSV",
    "HDM05": "JSON",
    "HowTo100M": "JSON",
    "HumanAct12": "NPY",
    "imagenet100": "FOLDER",
    "Imagenet21k": "FOLDER/PTH",
    "imagenetood": "FOLDER/PTH",
    "IP102": "TXT",
    "Madverse": "JSON",
    "MammalNet": "-",
    "Mapillary_Vistas": "JSON",
    "marine-tree": "CSV",
    "Matador": "WEB",
    "CheXpert": "CSV",
    "DeepLesion": "CSV",
    "MIMIC-CXR": "CSV",
    "OpenCell": "CSV",
    "MillionAid": "XML",
    "mini_kinetics": "JSON",
    "Mintrec": "-",
    "moments_in_time": "JSON",
    "NABirds": "-",
    "Objects365": "WEB",
    "openimages": "JSON",
    "OpenLoris": "JSON",
    "PACO": "JSON",
    "Pascal": "XML",
    "Portrait-Mode-400": "-",
    "PseudoAdverbs": "CSV",
    "rare_species": "CSV",
    "SomethingSomethingV2": "JSON",
    "StanfordOnlineProducts": "FOLDER",
    "stanford_cars": "FOLDER",
    "SUN360": "WEB",
    "SUN397": "CSV",
    "SUN908": "WEB",
    "TreeOfLife": "CSV",
    "UCF101": "CSV",
    "vegfru": "JSON",
    "Visual_Genome": "JSON"
}


dataset_with_star = set(["ADE20K", "Imagenet21k", "imagenetood", "SomethingSomethingV2"])

def get_dataset_name(dataset_key: str, file_path: str) -> str:
    """
    Get the dataset name from the dataset key.
    """
    dataset_name = dataset_names.get(dataset_key, dataset_key.replace("_", " "))
    last_folder = os.path.basename(os.path.dirname(file_path))
    if dataset_key in dataset_with_star:
        dataset_name += "$^*$"
    if dataset_key != last_folder:
        dataset_name += f" ({last_folder.replace('_', ' ')})"
    elif len(os.listdir(os.path.dirname(file_path))) > 1:
        dataset_name += f" ({os.path.basename(file_path).replace('.json', '').replace('_', ' ').replace('hierarchy', '')})"
    return dataset_name


def get_dataset_stats(dataset_key, file_path) -> pd.DataFrame:
    try:
        G, root_id = load_graph_from_file(file_path)
    except Exception as e:
        logger.warning(f"File {file_path} does not contain a hierarchy. Skipping...")
        print(e)
        raise e
    # statistics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_classes = len([n for n in G.nodes() if G.out_degree(n) == 0])
    max_depth = max(nx.single_source_shortest_path_length(G, root_id).values())

    # create a DataFrame with the statistics
    stats = {
        "Tasks": dataset_tasks.get(dataset_key),
        "Dataset": get_dataset_name(dataset_key, file_path),
        "Hierarchy Source": hierarchy_sources.get(dataset_key, "-"),
        "Original Format": original_format.get(dataset_key, "-"),
        "Nodes": num_nodes,
        "Edges": num_edges,
        "Depth": max_depth,
        "Classes": num_classes,
    }

    df = pd.DataFrame([stats])

    return df


def get_final_table(datasets:List[pd.DataFrame], save_path:str=None) -> str:

    table_df = pd.concat(datasets, ignore_index=False)
    table_df.set_index(["Tasks", "Dataset"], inplace=True)
    table_df.sort_index(inplace=True)
    table_df.reset_index("Dataset", inplace=True)

    # hide rows containing val or test in the dataset name
    table_df = table_df[~table_df["Dataset"].str.contains("val|test", case=False, na=False)]

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
    table_df.columns = list(map(lambda x: f"\\textbf{{{x}}}", table_df.columns))
    caption = "All hierarchies available currently in \\emph{HierVision}. Datasets with multiple hierarchy versions (e.g., coarse/fine) are marked with $^*$. If the hierarchy was sourced from another paper, it is cited in the “Hierarchy Source” column."
    label = "tab:hierarchy_coverage"

    table_latex = table_df.to_latex(
        column_format="lll|rrrr",
        index=True,
        multirow=True,
        multicolumn=True,
        escape=False,  # to allow for special characters
        bold_rows=False,
        longtable=False,
        caption=caption,
        label=label,
    )

    print(table_latex)

    if save_path:
        with open(save_path, "w") as f:
            f.write(table_latex)
        logger.info(f"Table saved to {save_path}")


if __name__ == "__main__":

    path_to_hierarchies = "code/HierVision/hierarchies/**/{dataset_name}/**/*.json"
    
    datasets = []
    for dataset_folder in dataset_names.keys():
        file_path = path_to_hierarchies.format(dataset_name=dataset_folder)
        file_path = glob(file_path, recursive=True)  # Get the first matching file
        for path in file_path:
            try:
                stats_df = get_dataset_stats(dataset_folder, path)
                datasets.append(stats_df)
            except Exception as e:
                logger.error(f"Failed to process dataset {path}: {e}")
                continue
    
    if datasets:
        get_final_table(datasets, save_path="code/HierVision/hierarchy_stats.tex")
    else:
        logger.warning("No datasets found or all datasets failed to process.")
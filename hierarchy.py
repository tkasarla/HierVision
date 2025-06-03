import argparse
from utils.graph_utils import load_graph_from_file

def main(args):
    hierarchy, root_id = load_graph_from_file(args.hierarchy_file)
    print(f"Hierarchy loaded from {args.hierarchy_file}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--hierarchy_file", dest="hierarchy_file", type=str, required=True)
    args = parser.parse_args()

    main(args)
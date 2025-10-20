# HierVision

Welcome to this repository of standard and reproducible hierarchies for vision datasets.

All hierarchies are formatted as a json file in the hierarchies folder, with the following format:
        {
          "directed": true,
          "multigraph": false,
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

## Load hierarchy
Hierarchies can be loaded using util functions in the utils folder, or by providing the json hierarchy path to the Hierarchy class like so:
'''
from utils import Hierarchy, HierarchyCatalog
hierarchy = Hierarchy('hierarchy_json_filepath.json')
'''
This loads the hierarchy for the dataset as a graph.

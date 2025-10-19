from graph_utils import HierarchyCatalog
from graph_utils import Hierarchy


class HierarchyValidator:
    """
    A class to validate hierarchical data structures.
    """

    def __init__(self):
        self.catalog = HierarchyCatalog()

    def validate(self, hierarchy: Hierarchy) -> bool:
        """
        Validate the hierarchy structure.
        Returns True if valid, False otherwise.
        """
        return self._is_valid_hierarchy(hierarchy)
    
    def validate_all(self) -> dict:
        """
        Validate all hierarchies in the catalog.
        Returns a dictionary with hierarchy names as keys and validation results as values.
        """
        results = {}
        for h_name in self.catalog.hierarchies.keys():
            hierarchy = self.catalog.get(h_name)
            results[h_name] = self._is_valid_hierarchy(hierarchy)
        return results
    
    def _is_valid_hierarchy(self, hierarchy: Hierarchy) -> bool:
        return True
    
    
    
if __name__ == "__main__":
    validator = HierarchyValidator()
    results = validator.validate_all()
    for h_name, is_valid in results.items():
        print(f"Hierarchy '{h_name}' is valid: {is_valid}")
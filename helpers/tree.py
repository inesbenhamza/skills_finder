class TreeNode:
    def __init__(self, name: str, technology_name, parent=None, level=0):
        self.name = name
        self.children = {}
        self.parent: TreeNode = parent
        self.level = level
        self.technology_name = technology_name
       
    def add_child(self, words, technology_name):
        if not words:
            return
        words = [word.lower() for word in words]
        first_word = words[0]
        if first_word not in self.children:
            self.children[first_word] = TreeNode(first_word, technology_name, parent=self, level=self.level+1)
        if len(words) > 1:
            self.children[first_word].add_child(words[1:], technology_name)
    
    def get_parents(self, include_full_path=False):
        """
        Retrieve parent names.
        
        Args:
            include_full_path (bool): If True, returns a list of full path names.
                                      If False, returns a list of immediate parent names.
        
        Returns:
            list: List of parent names
        """
        parents = []
        current = self.parent
        
        while current is not None:
            # Clean parent name (remove <eos/> and parentheses)
            clean_name = current.name.removesuffix(' <eos/>').replace('(', '').replace(')', '')
            
            if include_full_path:
                # If full path is desired, add cumulative parents
                if not parents:
                    parents.append(clean_name)
                else:
                    parents.append(f"{parents[-1]} {clean_name}")
            else:
                # If just immediate parents, add the clean name
                parents.append(clean_name)
            
            current = current.parent
        
        # Reverse to get parents from root to current node
        return list(reversed(parents))
    
    def prune(self):
        while len(self.children) == 1:
            only_child = next(iter(self.children.values()))
            if '(' in only_child.name:
                return
            self.name += " " + only_child.name
            self.children = only_child.children
        for child in self.children.values():
            child.prune()
    
    def match_term(self, term):
        name_to_match = self.name.removesuffix(' <eos/>').replace('(', '').replace(')', '')
        
        # Get parents for matching
        parents = self.get_parents()
        if len(parents) == 0:
            parents_denomination = []
        else:
            parents_denomination = [parents[0]] + [" ".join(parents[:i+1]).strip() for i in range(1, len(parents))]
        
        names_to_match = [f"{p} {name_to_match}".strip() for p in parents_denomination]

        if term in names_to_match:
            raise Exception(self.technology_name)
        
        for child in self.children.values():
            if child.match_term(term)[0]:
                # return True, child.technology_name
                raise Exception(child.technology_name)
        
        return False, ''
    
    def display(self, level=0):
        if self.name == "<eos/>":
            return
        
        print("  " * self.level + self.name.removesuffix(' <eos/>').replace('(', '').replace(')', ''))
        for child in sorted(self.children.values(), key=lambda x: x.name):
            child.display()

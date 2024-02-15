from typing import List, Any

class ItemsDataset:
    
    def __init__(self, items: List[Any]):
        self.items = items
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, index):
        return self.items[index]
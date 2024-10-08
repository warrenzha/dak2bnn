"""
Adapted from https://github.com/treforevans/uci_datasets/blob/master/uci_datasets/dataset.py
"""

import os
import gzip
import torch
from torch.utils.data import Dataset

# save some global variables
# Identify the small, intermediate, and large datasets from Yang et al., 2016 by the
# name of the dataset reported in that paper.
small_datasets = [
    "challenger",
    "fertility",
    "slump",
    "automobile",
    "servo",
    "cancer",
    "hardware",
    "yacht",
    "autompg",
    "housing",
    "forest",
    "stock",
    "pendulum",
    "energy",
    "concrete",
    "solar",
    "airfoil",
    "wine",
    "obesity",
    "churn",
]
intermediate_datasets = [
    "gas",
    "skillcraft",
    "sml",
    "parkinsons",
    "pumadyn",
    "poletele",
    "elevators",
    "kin40k",
    "protein",
    "kegg",
    "keggu",
    "ctslice",
]
large_datasets = ["3droad", "song", "buzz", "electric"]
# also Identify all of the datasets, and specify the ( n_points, n_dimensions ) of each
# as a tuple.
all_datasets = {
    "3droad": (434874, 3),
    "autompg": (392, 7),
    "bike": (17379, 17),
    "challenger": (23, 4),
    "concreteslump": (103, 7),
    "energy": (768, 8),
    "forest": (517, 12),
    "houseelectric": (2049280, 11),
    "keggdirected": (48827, 20),
    "kin40k": (40000, 8),
    "parkinsons": (5875, 20),
    "pol": (15000, 26),
    "pumadyn32nm": (8192, 32),
    "slice": (53500, 385),
    "solar": (1066, 10),
    "stock": (536, 11),
    "yacht": (308, 6),
    "airfoil": (1503, 5),
    "autos": (159, 25),
    "breastcancer": (194, 33),
    "buzz": (583250, 77),
    "concrete": (1030, 8),
    "elevators": (16599, 18),
    "fertility": (100, 9),
    "gas": (2565, 128),
    "housing": (506, 13),
    "keggundirected": (63608, 27),
    "machine": (209, 7),
    "pendulum": (630, 9),
    "protein": (45730, 9),
    "servo": (167, 4),
    "skillcraft": (3338, 19),
    "sml": (4137, 26),
    "song": (515345, 90),
    "tamielectric": (45781, 3),
    "wine": (1599, 11),
    "obesity": (2111, 16),
    "churn": (3150, 13),
}


class UCIDatasets(Dataset):
    """
    Load UCI dataset.

    Args:
        dataset: name of the dataset to load.
        print_stats: if true then will print stats about the dataset.
    """
    
    def __init__(self, dataset: str, print_stats: bool = True, dtype = torch.float32, transform=None):
        assert isinstance(dataset, str), "dataset must be a string"
        dataset = dataset.lower().replace(" ", "").replace("_", "")
        

        id_map = {
            "slump": "concreteslump", "automobile": "autos", "cancer": "breastcancer",
            "hardware": "machine", "forestfires": "forest", "solarflare": "solar",
            "gassensor": "gas", "poletele": "pol", "kegg": "keggdirected",
            "keggu": "keggundirected", "ctslice": "slice", "electric": "houseelectric",
            "pumadyn": "pumadyn32nm"
        }
        if dataset in id_map:
            dataset = id_map[dataset]

        self.dtype = dtype
        self.transform = transform

        dirpath = os.path.dirname(__file__)
        dataset_dirpath = os.path.join(dirpath, "datasets")
        try:
            # self.test_mask = self.load_txt(os.path.join(dataset_dirpath, dataset, "test_mask.csv.gz"), dtype=bool)
            data = self.load_txt(os.path.join(dataset_dirpath, dataset, "data.csv.gz"), dtype=dtype)
            if dataset == "song":
                data1 = self.load_txt(os.path.join(dataset_dirpath, dataset, "data1.csv.gz"), dtype=dtype)
                data = torch.cat([data, data1], dim=0)
        except Exception as e:
            print("Load failed, maybe dataset string is not correct.")
            raise e

        self.x = data[:, :-1]
        self.y = data[:, -1]
        # self.y = data[:, -1].unsqueeze(1)

        if print_stats:
            print(f"{dataset} dataset, N={self.x.shape[0]}, d={self.x.shape[1]}")

        self.__name__ = str(dataset)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        input = self.x[idx]
        target = self.y[idx]

        # Apply the transform (if any) to the data
        if self.transform:
            input = self.transform(input)

        sample = (input, target)
        # sample = {'x': self.x[idx], 'y': self.y[idx]}
        return sample

    @property
    def tensors(self):
        return (self.x, self.y)

    def load_txt(self, filepath, dtype):
        with gzip.open(filepath, 'rt') as f:
            data = [line.strip().split(',') for line in f]
        data = torch.tensor([list(map(float, row)) for row in data], dtype=dtype)
        return data
from src.load_data.m3 import M3Dataset
from src.load_data.m4 import M4Dataset
from src.load_data.tourism import TourismDataset
from src.load_data.gluonts import GluontsDataset

DATASETS = {
    'M3': M3Dataset,
    'M4': M4Dataset,
    'Tourism': TourismDataset,
    'Gluonts': GluontsDataset,
}

DATA_GROUPS = [
    ('Gluonts', 'm1_monthly'),
    ('Gluonts', 'm1_quarterly'),
    ('M3', 'Monthly'),
    ('M3', 'Quarterly'),
    ('Tourism', 'Monthly'),
    ('Tourism', 'Quarterly'),
    ('M4', 'Monthly'),
    ('M4', 'Quarterly'),
]

GROUP_IDX = 2

import json
from tqdm import tqdm

def read_gazeta_records(file_name, amount):
    records = []
    with open(file_name, "r") as r:
        for line in tqdm(r, total=amount):
            records.append(json.loads(line))
    return records

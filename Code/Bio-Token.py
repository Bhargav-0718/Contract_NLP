import json
import os
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# ------------------------
# Paths
# ------------------------
full_json_path = "CUAD_v1/CUAD_v1.json"
processed_dir = "CUAD_v1/processed"
os.makedirs(processed_dir, exist_ok=True)

# ------------------------
# Load full dataset
# ------------------------
with open(full_json_path, "r", encoding="utf-8") as f:
    cuad_data = json.load(f)["data"]

print(f"Total contracts in dataset: {len(cuad_data)}")

# ------------------------
# Split dataset
# ------------------------
train_val_test = train_test_split(cuad_data, test_size=0.3, random_state=42)
train_data, temp_data = train_val_test
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

splits = {
    "train": train_data,
    "validation": val_data,
    "test": test_data
}

print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

# ------------------------
# Helper function to flatten paragraphs into one row per clause
# ------------------------
def flatten_paragraphs(contract):
    """
    Returns a list of dicts with:
    paragraph -> full paragraph text
    clause    -> answer text
    label     -> clause type
    """
    rows = []
    for para in contract.get("paragraphs", []):
        para_text = para.get("context") or para.get("text") or ""
        if not para_text.strip():
            continue

        for qa in para.get("qas", []):
            # Extract label from question
            label = qa["question"].split('"')[1] if '"' in qa["question"] else "CLAUSE"

            for ans in qa.get("answers", []):
                rows.append({
                    "paragraph": para_text,
                    "clause": ans["text"],
                    "label": label
                })
    return rows

# ------------------------
# Process each split
# ------------------------
for split_name, split_data in splits.items():
    csv_file = os.path.join(processed_dir, f"{split_name}_clauses.csv")
    all_rows = []

    for contract in tqdm(split_data, desc=f"Processing {split_name}"):
        all_rows.extend(flatten_paragraphs(contract))

    # Save CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["paragraph", "clause", "label"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"{split_name} -> CSV file: {csv_file}, total rows: {len(all_rows)}")

print("Processing complete! CSV files with paragraph, clause, and label are ready for LegalBERT.")

import json
import codecs

def load_dataset(file_path):
    """Loads a .jsonl or multi-line .json file where each line is a JSON object."""
    data = []
    with codecs.open(file_path, 'r', 'utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from line in {file_path}. Skipping.")
    return data
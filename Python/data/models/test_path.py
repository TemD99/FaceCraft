import json

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print("JSON data loaded successfully.")
            return data
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

# Path to your JSON file
json_path = '/data/PROCESSED_DATASET/TESTDATASET/dataset.json'
data = load_json(json_path)

if data:
    # Print first few entries to verify
    print(data['labels'][:10])


if __name__ == "__main__":
    file_path = '/data/PROCESSED_DATASET/TESTDATASET/dataset.json'
    data = load_json(file_path)
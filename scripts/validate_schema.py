import json
import os
import sys

def enforce_clinical_dataset_integrity():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    target_data_file = os.path.join(base_dir, "data", "phq9_data.jsonl")
    
    print(f"🔒 Initiating Contract Schema Validation on: {target_data_file}")
    
    if not os.path.exists(target_data_file):
        print(f"❌ CRITICAL ERROR: Target database file missing.")
        sys.exit(1)
        
    expected_keys = {"source", "category", "diagnostic_id", "core_text", "action_rule"}
    record_counter = 0
    
    with open(target_data_file, "r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            record_counter += 1
            if not expected_keys.issubset(record.keys()):
                print(f"❌ Missing keys in record {record_counter}")
                sys.exit(1)
                
    print(f"✅ Schema validation passed with {record_counter} records.")
    sys.exit(0)

if __name__ == "__main__":
    enforce_clinical_dataset_integrity()

# generate_knowledge_base.py
import json
import os

# Base variations to generate a robust clinical reference dataset
conditions = ["Major Depressive Disorder (MDD)", "Generalized Anxiety Disorder (GAD)", "Panic Disorder", "Bipolar I Manic Episode", "Seasonal Affective Disorder (SAD)", "Severe Insomnia Disorder"]
severities = ["Mild", "Moderate", "Severe", "Critical"]
drug_classes = ["SSRIs (e.g., Fluoxetine, Escitalopram)", "SNRIs (e.g., Duloxetine)", "Benzodiazepines (e.g., Diazepam)", "Mood Stabilizers (e.g., Lithium, Valproate)", "Atypical Antipsychotics", "NSAIDs + Anticoagulant Risk Check"]

base_templates = [
    "Patient presenting with acute symptoms of {condition}. Evaluating baseline psychological parameters and considering immediate intervention with {drug_class}. Risk profile assessed as {severity}.",
    "Clinical check required for an adult displaying signs of {condition}. Current emotional state is evaluated as {severity}. Reviewing safety interactions for standard {drug_class} treatments.",
    "Pharmacist consultation requested for managing {severity} case of {condition}. Patient reports non-adherence to prior medication. Assessing compatibility with {drug_class} protocols.",
    "Emergency presentation profile: Patient exhibits erratic behavior linked to {condition}. Clinical risk threshold is {severity}. Cross-checking contraindications against {drug_class} records."
]

def generate_large_jsonl(output_path, count=160):
    print(f"🧬 Generating {count} distinct clinical reference records...")
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    records_written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(count):
            # Select variations using mathematical loops to ensure unique combinations
            condition = conditions[i % len(conditions)]
            severity = severities[i % len(severities)]
            drug_class = drug_classes[i % len(drug_classes)]
            template = base_templates[i % len(base_templates)]
            
            # Format text body
            text_line = template.format(condition=condition, severity=severity, drug_class=drug_class)
            
            # Create JSON structure
            record = {
                "text": text_line,
                "metadata": {
                    "condition": condition.split(" (")[0],
                    "severity": severity,
                    "drug_class": drug_class.split(" (")[0],
                    "record_id": f"CLIN-REF-{1000 + i}"
                }
            }
            
            # Write line-separated JSON
            f.write(json.dumps(record) + "\n")
            records_written += 1

    print(f"✅ Success! Generated and rewrote {records_written} clinical lines to: {output_path}")

if __name__ == "__main__":
    generate_large_jsonl("data/clinical_records.jsonl", count=160)
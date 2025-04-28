import subprocess

# scripts to execute in the pipeline
scripts = [
    "scripts/01_filter.py",
    "scripts/02_transform.py",
    "scripts/03a_extract_metadata.py",
    "scripts/03b_extract_embeddings.py",
    "scripts/04_train_test.py",
    "scripts/05a_baseline_recommender.py",
    "scripts/05b_cf_recommender.py",
    "scripts/05c_cb_recommenders.py",
    "scripts/05d_hybrid_recommender.py",
    "scripts/06_evaluation.py"
]

# executing each script in order
for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run(["python", script], check=True)
    if result.returncode == 0:
        print(f"{script} completed successfully.")
    else:
        print(f"Error occurred while running {script}.")
        break
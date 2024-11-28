import subprocess

# Define the path to your file
# file_path = "standard_decision_tree/DecisionTreeV1.py"
# file_path = "utility_based_classification/ubc_main.py"
file_path = "OPM_approach/ubo_main.py"

# Run the file 10 times in a row
for i in range(10):
    print(f"Running iteration {i + 1}...")
    try:
        # Execute the Python file
        subprocess.run(["python", file_path], check=True)
        print(f"Iteration {i + 1} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred during iteration {i + 1}: {e}\n")

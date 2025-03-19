"""
This script automates the execution of multiple model scripts and records their performance metrics.
It runs each script, captures the total execution time and response generation time, and writes these metrics to an Excel file.
The script also checks if CUDA is available and records whether each model ran on a CPU or GPU.
"""
# Import necessary libraries
import subprocess # Library for running shell commands
import pandas as pd # Library for data manipulation and analysis

# List of scripts to run
scripts = [
        "hf_transformers_causallm_gemma2b_one_cpu.py",
        "hf_transformers_causallm_gemma2b_multiple_cpu.py",
        "hf_transformers_causallm_gemma2b_gpu_4bit.py",
        "hf_transformers_causallm_gemma2b_gpu_bfloat16.py",
        "hf_transformers_causallm_gemma2b_gpu_float16.py",
        "hf_transformers_causallm_gemma2b_gpu_int8.py",
        "hf_transformers_causallm_gemma2b_gpu.py",
        "llama_gguf_gemma2b.py",
        "hf_transformers_pipeline_gemma2b.py"
]



# Initialize a list to store the results
results = []

# Run each script and capture the output
for script in scripts:
    try:
        # Run the script and capture the output
        result = subprocess.run(["python", script], capture_output=True, text=True)
        output = result.stdout
        print(f"Output from {script}:\n{output}")

        # Extract the relevant information from the output
        model_name = None
        device = None
        total_time = None
        response_time = None
        param_size = None
        file_size = None
        for line in output.split("\n"):
            if "Model name" in line:
                model_name = line.split(":")[1].strip()
            if "Device" in line:
                device = line.split(":")[1].strip()
            if "Total execution time" in line:
                total_time = float(line.split(":")[1].strip().split()[0])
            if "Response generation time" in line:
                response_time = float(line.split(":")[1].strip().split()[0])
            if "Parameter size" in line:
                param_size = int(line.split(":")[1].strip())
            if "File size" in line:
                file_size = int(line.split(":")[1].strip())
        
        # Convert parameter size to millions and file size to gigabytes
        param_size_billions = f"{param_size / 1_000_000_000:.2f}" if param_size is not None else None
        file_size_gb = f"{file_size / 1_000_000_000:.2f}" if file_size is not None else None

        # Append the result to the DataFrame
        new_data = {
            "Script": script,
            "Model": model_name,
            "Device": device,
            "Total Execution Time (s)": total_time,
            "Response Generation Time (s)": response_time,
            "Parameter Size (Billions)": param_size_billions,
            "File Size (GB)": file_size_gb
        }
        
        # Append new data to the results list
        results.append(new_data)
        #print(f"Appended data for {script}: {new_data}")
        
    except Exception as e:
        print(f"Error running script {script}: {e}")

# Create a DataFrame from the results
df_updated = pd.DataFrame(results)

# Write the DataFrame to the Excel file (overwrite if exists)
df_updated.to_excel("model_performance.xlsx", index=False)

# Print a message indicating that the results have been written to the Excel file
print("Results written to model_performance.xlsx")
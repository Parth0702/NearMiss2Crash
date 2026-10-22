import subprocess
import os

# Run the evaluation script
result = subprocess.run([
    r"D:\SC project\.venv\Scripts\python.exe",
    r"D:\SC project\experiments\distilbert_extension\run_common_test_eval.py"
], capture_output=True, text=True, timeout=600)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")

# Save output
with open(r"D:\SC project\experiments\distilbert_extension\eval_log.txt", "w") as f:
    f.write(result.stdout)
    f.write(result.stderr)
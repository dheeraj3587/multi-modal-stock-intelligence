
import subprocess
import sys

def run_flake8(path):
    """
    Runs flake8 on the specified path and returns the output and return code.

    Args:
        path (str): The file or directory path to check with flake8.

    Returns:
        tuple: A tuple containing:
            - str: The standard output from flake8.
            - int: The return code of the flake8 process.
    """
    try:
        process = subprocess.run(
            ['flake8', path],
            capture_output=True,
            text=True,
            check=False
        )
        return process.stdout, process.returncode
    except FileNotFoundError:
        return "Error: flake8 command not found. Make sure flake8 is installed and in your PATH.", 127
    except Exception as e:
        return f"An unexpected error occurred: {e}", 1

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_flake8.py <path>")
        sys.exit(1)
    
    path_to_check = sys.argv[1]
    output, return_code = run_flake8(path_to_check)

    if output:
        print(output)

    if return_code != 0:
        print(f"Flake8 found issues. Return code: {return_code}")
    else:
        print("Flake8 found no issues.")

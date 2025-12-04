import subprocess
import numpy as np
from pathlib import Path

def run_cpp_pca(
    executable: str = "./pca",
    a_dat: str = "A.dat",
    output_scores: str = "PC_scores.dat",
    cwd: str = ".",
    timeout: float = 30.0
):
    """
    Run the C++ PCA program and return the principal component scores as a NumPy array.
    
    Parameters
    ----------
    executable : str
        Path to the compiled pca binary (default: "./pca")
    a_dat : str
        Input data file (must exist in cwd)
    output_scores : str
        Output file written by C++ program
    cwd : str
        Working directory where files are located
    timeout : float
        Timeout for subprocess in seconds
    
    Returns
    -------
    dict
        Contains:
        - scores: np.ndarray (n_samples, n_components)
        - explained_variance_ratio: np.ndarray or None
        - singular_values: np.ndarray or None
        - loadings: None (not parsed)
        - stdout: str (full console output)
    """
    # Resolve paths
    exec_path = Path(cwd) / executable
    a_path = Path(cwd) / a_dat
    scores_path = Path(cwd) / output_scores

    # Check existence
    if not exec_path.exists():
        raise FileNotFoundError(f"Executable not found: {exec_path}")
    if not a_path.exists():
        raise FileNotFoundError(f"Input file not found: {a_path}")

    # Use absolute path to avoid shell relative path issues
    abs_exec_path = exec_path.resolve()

    # Run the C++ program
    try:
        result = subprocess.run(
            str(abs_exec_path),
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"C++ PCA program failed with return code {e.returncode}\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}"
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError("C++ PCA program timed out")

    stdout = result.stdout
    stderr = result.stderr

    if stderr:
        print("Warning: C++ program emitted to stderr:\n", stderr)

    # Check output file
    if not scores_path.exists():
        raise FileNotFoundError(f"Expected output file not created: {scores_path}")

    # Parse scores
    scores = parse_matrix_market_dense(scores_path)

    # Parse additional info from stdout
    explained_variance_ratio = parse_explained_variance_from_stdout(stdout)
    singular_values = parse_singular_values_from_stdout(stdout)
    loadings = None  # Not currently parsed from output

    return {
        "scores": scores,
        "explained_variance_ratio": explained_variance_ratio,
        "singular_values": singular_values,
        "loadings": loadings,
        "stdout": stdout
    }


def parse_matrix_market_dense(filepath):
    """
    Parse a dense MatrixMarket coordinate file (like PC_scores.dat) into a NumPy array.
    Assumes real general format.
    """
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith('%') and line.strip()]

    if not lines:
        raise ValueError("Empty or invalid MatrixMarket file")

    # First non-comment line: rows cols nnz
    rows, cols, _ = map(int, lines[0].split())
    data = np.zeros((rows, cols))

    for line in lines[1:]:
        if not line:
            continue
        parts = line.split()
        if len(parts) != 3:
            continue  # Skip malformed lines
        r, c, val = parts
        r_idx = int(r) - 1
        c_idx = int(c) - 1
        data[r_idx, c_idx] = float(val)

    return data


def parse_explained_variance_from_stdout(stdout):
    """Extract explained_variance_ratio from C++ stdout if present"""
    for line in stdout.splitlines():
        if "Explained variance ratio:" in line:
            parts = line.split(":", 1)[1].strip().split()
            return np.array([float(x) for x in parts])
    return None


def parse_singular_values_from_stdout(stdout):
    """Extract singular values from C++ stdout"""
    for line in stdout.splitlines():
        if "Singular values:" in line:
            parts = line.split(":", 1)[1].strip().split()
            return np.array([float(x) for x in parts])
    return None


# ==================== Example Usage ====================
if __name__ == "__main__":
    try:
        result = run_cpp_pca(
            executable="./pca",
            a_dat="A.dat",
            output_scores="PC_scores.dat",
            cwd="."
        )

        print("PCA completed successfully!")
        print("Scores shape:", result["scores"].shape)
        print("\nPrincipal Component Scores:\n", result["scores"])

        if result["explained_variance_ratio"] is not None:
            print("\nExplained Variance Ratio:", result["explained_variance_ratio"])
            print("Cumulative:", np.cumsum(result["explained_variance_ratio"]))

        if result["singular_values"] is not None:
            print("\nSingular Values:", result["singular_values"])

    except Exception as e:
        print("Error:", e)

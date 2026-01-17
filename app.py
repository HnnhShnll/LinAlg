from flask import Flask, request, jsonify, render_template
import numpy as np

# Initialize the Flask application
# This creates the server instance that listens for web traffic.
app = Flask(__name__)

# =========================================================
# TEAMMATE'S CORE MATH LOGIC: RREF ENGINE
# =========================================================

def rref(matrix, tol=1e-10):
    """
    Computes the Reduced Row Echelon Form (RREF) of a matrix.
    This is the manual Gaussian Elimination process required to 
    solve for the null space of (A - λI).
    """
    m, n = matrix.shape
    A = matrix.copy().astype(float)
    pivots = []
    row = 0
    
    for col in range(n):
        # 1. Search for a pivot element in the current column
        pivot_row = None
        for i in range(row, m):
            if abs(A[i, col]) > tol:
                pivot_row = i
                break
        
        # 2. If no valid pivot exists, skip this column (free variable)
        if pivot_row is None:
            continue
        
        # 3. Mark this column as a pivot column
        pivots.append(col)
        
        # 4. Swap current row with the row containing the pivot
        if pivot_row != row:
            A[[row, pivot_row]] = A[[pivot_row, row]]
        
        # 5. Normalize the pivot row so the leading entry is exactly 1
        A[row] = A[row] / A[row, col]
        
        # 6. Eliminate all other entries in this column to create zeros
        for i in range(m):
            if i != row and abs(A[i, col]) > tol:
                A[i] = A[i] - A[i, col] * A[row]
        
        # 7. Move to the next row for the next pivot
        row += 1
    
    # Final cleanup: force very small floating point numbers to absolute zero
    A[np.abs(A) < tol] = 0
    return A, pivots

# =========================================================
# WEB SERVER ROUTES
# =========================================================

@app.route('/')
def index():
    """
    THE HOME ROUTE: 
    This is what the browser calls when you first visit the URL.
    It links the Python logic to your index.html file.
    """
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    """
    THE COMPUTATION API:
    Receives matrix data from the website via JSON, calculates
    eigenvalues and basis vectors, and sends the results back.
    """
    try:
        # Step 1: Parse the incoming JSON data
        data = request.json
        matrix_data = data.get('matrix')
        matrix = np.array(matrix_data, dtype=float)
        
        # Step 2: Input Validation
        if matrix.shape[0] != matrix.shape[1]:
            return jsonify({"error": "Matrix must be square"}), 400
        
        if matrix.shape[0] > 5:
            return jsonify({"error": "Matrix size must be ≤ 5 for performance"}), 400
        
        # Step 3: Find Eigenvalues using NumPy's characterstic equation solver
        eigvals = np.linalg.eigvals(matrix)
        
        # Step 4: Filter for Real Eigenvalues only
        # Most linear algebra 101 courses focus on real-valued eigenspaces.
        real_eigvals = [v.real for v in eigvals if abs(v.imag) < 1e-10]
        
        # Step 5: Group and unique-ify the eigenvalues
        unique_vals = []
        for val in real_eigvals:
            if not any(abs(val - u) < 1e-8 for u in unique_vals):
                unique_vals.append(val)
        
        results = []
        
        # Step 6: Solve for the Eigenspace Basis of each unique eigenvalue
        for eigval in unique_vals:
            n = matrix.shape[0]
            
            # Construct the characteristic matrix (A - λI)
            A_lambda = matrix - eigval * np.eye(n)
            
            # Use the RREF function to find the pivot and free columns
            rref_matrix, pivots = rref(A_lambda)
            
            # Free variables identify the dimensions of the eigenspace
            free_vars = [c for c in range(n) if c not in pivots]
            
            # Build the basis vectors from the RREF coefficients
            basis = []
            for free_var in free_vars:
                vec = np.zeros(n)
                vec[free_var] = 1 # The free variable position
                
                for i, pivot_col in enumerate(pivots):
                    if i < len(rref_matrix):
                        # Leading variables are expressed in terms of free variables
                        vec[pivot_col] = -rref_matrix[i, free_var]
                
                basis.append(vec)
            
            # Step 7: Clean and format the vectors for JSON transport
            formatted_basis = []
            for vec in basis:
                # Ensure we aren't returning a trivial zero vector
                if not np.allclose(vec, 0, atol=1e-8):
                    formatted_vec = []
                    for x in vec:
                        # Clean up rounding errors (0.9999 -> 1)
                        if abs(x - round(x)) < 1e-8:
                            formatted_vec.append(int(round(x)))
                        else:
                            formatted_vec.append(float(f"{x:.4f}"))
                    formatted_basis.append(formatted_vec)
            
            # Step 8: Clean and format the eigenvalue itself
            if abs(eigval - round(eigval)) < 1e-8:
                formatted_eigval = int(round(eigval))
            else:
                formatted_eigval = float(f"{eigval:.4f}")
            
            # Step 9: Package results for this eigenvalue
            results.append({
                "eigenvalue": formatted_eigval,
                "geometric_multiplicity": len(formatted_basis),
                "basis": formatted_basis
            })
        
        # Step 10: Final JSON response to the browser
        return jsonify({
            "matrix_size": matrix.shape[0],
            "real_eigenvalues_count": len(unique_vals),
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400



if __name__ == '__main__':
    app.run(debug=True)

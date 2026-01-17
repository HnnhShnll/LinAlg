from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

def rref(matrix, tol=1e-10):
    """Compute RREF of matrix."""
    m, n = matrix.shape
    A = matrix.copy().astype(float)
    pivots = []
    row = 0
    
    for col in range(n):
        # Find pivot
        pivot_row = None
        for i in range(row, m):
            if abs(A[i, col]) > tol:
                pivot_row = i
                break
        
        if pivot_row is None:
            continue
        
        pivots.append(col)
        
        # Swap rows
        if pivot_row != row:
            A[[row, pivot_row]] = A[[pivot_row, row]]
        
        # Normalize
        A[row] = A[row] / A[row, col]
        
        # Eliminate
        for i in range(m):
            if i != row and abs(A[i, col]) > tol:
                A[i] = A[i] - A[i, col] * A[row]
        
        row += 1
    
    A[np.abs(A) < tol] = 0
    return A, pivots

@app.route('/compute', methods=['POST'])
def compute():
    """Compute eigenvalues and eigenspace bases."""
    try:
        data = request.json
        matrix_data = data.get('matrix')
        matrix = np.array(matrix_data, dtype=float)
        
        # Validation
        if matrix.shape[0] != matrix.shape[1]:
            return jsonify({"error": "Matrix must be square"}), 400
        
        if matrix.shape[0] > 5:
            return jsonify({"error": "Matrix size must be ≤ 5"}), 400
        
        # Get eigenvalues
        eigvals = np.linalg.eigvals(matrix)
        real_eigvals = [v.real for v in eigvals if abs(v.imag) < 1e-10]
        
        # Group unique eigenvalues
        unique_vals = []
        for val in real_eigvals:
            if not any(abs(val - u) < 1e-8 for u in unique_vals):
                unique_vals.append(val)
        
        results = []
        
        for eigval in unique_vals:
            n = matrix.shape[0]
            
            # Form A - λI
            A_lambda = matrix - eigval * np.eye(n)
            
            # Get RREF
            rref_matrix, pivots = rref(A_lambda)
            
            # Find free variables
            free_vars = [c for c in range(n) if c not in pivots]
            
            # Build basis
            basis = []
            for free_var in free_vars:
                vec = np.zeros(n)
                vec[free_var] = 1
                
                for i, pivot_col in enumerate(pivots):
                    if i < len(rref_matrix):
                        vec[pivot_col] = -rref_matrix[i, free_var]
                
                basis.append(vec)
            
            # Filter out zero vectors and format
            formatted_basis = []
            for vec in basis:
                if not np.allclose(vec, 0, atol=1e-8):
                    # Format vector
                    formatted_vec = []
                    for x in vec:
                        if abs(x - round(x)) < 1e-8:
                            formatted_vec.append(int(round(x)))
                        else:
                            formatted_vec.append(float(f"{x:.4f}"))
                    formatted_basis.append(formatted_vec)
            
            # Format eigenvalue
            if abs(eigval - round(eigval)) < 1e-8:
                formatted_eigval = int(round(eigval))
            else:
                formatted_eigval = float(f"{eigval:.4f}")
            
            results.append({
                "eigenvalue": formatted_eigval,
                "geometric_multiplicity": len(formatted_basis),
                "basis": formatted_basis
            })
        
        return jsonify({
            "matrix_size": matrix.shape[0],
            "real_eigenvalues_count": len(unique_vals),
            "results": results
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

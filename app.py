from flask import Flask, request, jsonify, send_from_directory # Added send_from_directory
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__, static_folder='.') # Tell Flask to look in the current folder
CORS(app)

# ADD THIS ROUTE:
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/compute', methods=['POST'])
def compute():
    try:
        data = request.json
        matrix_data = data.get('matrix')
        
        # Convert to NumPy array
        matrix = np.array(matrix_data, dtype=float)
        
        if matrix.shape[0] != matrix.shape[1]:
            return jsonify({"error": "Matrix must be square"}), 400
        
        # Calculate Eigenvalues and Eigenvectors
        # eigenvalues: 1D array
        # eigenvectors: 2D array where columns are the vectors
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        results = []
        for i in range(len(eigenvalues)):
            results.append({
                "eigenvalue": round(complex(eigenvalues[i]).real, 4), 
                "eigenvector": [round(val, 4) for val in eigenvectors[:, i]]
            })
            
        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np

app = Flask(__name__, static_folder='static')
CORS(app)

# Serve main page
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# Compute eigenvalues/eigenvectors
@app.route('/compute', methods=['POST'])
def compute():
    try:
        data = request.get_json(force=True)
        if not data or 'matrix' not in data:
            return jsonify({"error": "No matrix provided"}), 400

        matrix_data = data['matrix']
        matrix = np.array(matrix_data, dtype=float)

        if matrix.shape[0] != matrix.shape[1]:
            return jsonify({"error": "Matrix must be square"}), 400

        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        results = []

        for i in range(len(eigenvalues)):
            val = eigenvalues[i]
            results.append({
                # Include imaginary part for complex eigenvalues
                "eigenvalue": [round(val.real, 4), round(val.imag, 4)],
                "eigenvector": [round(x, 4) for x in eigenvectors[:, i]]
            })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)

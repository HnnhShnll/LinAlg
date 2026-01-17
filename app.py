from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/compute", methods=["POST"])
def compute():
    try:
        data = request.json
        matrix_data = data.get("matrix")

        matrix = np.array(matrix_data, dtype=float)

        if matrix.shape[0] != matrix.shape[1]:
            return jsonify({"error": "Matrix must be square"}), 400

        eigenvalues, eigenvectors = np.linalg.eig(matrix)

        results = []
        for i in range(len(eigenvalues)):
            results.append({
                "eigenvalue": round(float(eigenvalues[i].real), 4),
                "eigenvector": [round(float(val), 4) for val in eigenvectors[:, i]]
            })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run()

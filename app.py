from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compute', methods=['POST'])
def compute():
    try:
        data = request.json
        matrix_data = data.get('matrix')
        matrix = np.array(matrix_data, dtype=float)
        
        if matrix.shape[0] != matrix.shape[1]:
            return jsonify({"error": "Matrix must be square"}), 400
        
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        
        results = []
        for i in range(len(eigenvalues)):
            ev = eigenvalues[i]
            
            if abs(ev.imag) < 1e-10:
                ev_formatted = f"{round(ev.real, 4)}"
            else:
                sign = "+" if ev.imag > 0 else "-"
                ev_formatted = f"{round(ev.real, 4)} {sign} {round(abs(ev.imag), 4)}i"

            vec = eigenvectors[:, i]
            scale_factor = 1
            for val in vec:
                if abs(val) > 1e-10:
                    scale_factor = val
                    break
            
            scaled_vec = vec / scale_factor
            formatted_vec = [round(v.real, 4) for v in scaled_vec]

            results.append({
                "eigenvalue": ev_formatted,
                "eigenvector": formatted_vec
            })
            
        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

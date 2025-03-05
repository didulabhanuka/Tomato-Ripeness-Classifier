from flask import Flask, request, jsonify, send_file
import os
import torch
import yaml
from PIL import Image
from collections import defaultdict
from models.yolo_model import yolo_model

# Initialize Flask App
app = Flask(__name__)

# Ensure prediction folder exists
os.makedirs("static/predictions", exist_ok=True)

# Load Class Names from YAML
yaml_path = "config.yaml"
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)
class_names = config["names"]

# Class grouping based on YAML names
category_mapping = {
    class_names[2]: "unripe",  # b_green
    class_names[5]: "unripe",  # l_green
    class_names[1]: "half-ripe",  # b_half_ripened
    class_names[4]: "half-ripe",  # l_half_ripened
    class_names[0]: "ripe",  # b_fully_ripened
    class_names[3]: "ripe"   # l_fully_ripened
}

def calculate_percentage(counts):
    """Calculate percentage of tomatoes in each ripeness stage."""
    total_tomatoes = sum(counts.values())
    
    # Ensure all categories exist in the output
    percentages = {
        "unripe": round((counts.get("unripe", 0) / total_tomatoes) * 100, 2) if total_tomatoes > 0 else 0.0,
        "half-ripe": round((counts.get("half-ripe", 0) / total_tomatoes) * 100, 2) if total_tomatoes > 0 else 0.0,
        "ripe": round((counts.get("ripe", 0) / total_tomatoes) * 100, 2) if total_tomatoes > 0 else 0.0
    }
    
    return percentages

def environmental_recommendations(percentages):
    """Calculate recommended temperature, light intensity, and humidity based on ripeness distribution."""
    
    # Temperature Ranges (°C)
    T_unripe, T_half_ripe, T_ripe = 20, 22, 24  # Midpoints of each range
    T_set = round((percentages["unripe"] * T_unripe + percentages["half-ripe"] * T_half_ripe + percentages["ripe"] * T_ripe) / 100, 2)

    # Light Intensity (lux)
    L_unripe, L_half_ripe, L_ripe = 7000, 5000, 3000  # Midpoints of each range
    L_set = round((percentages["unripe"] * L_unripe + percentages["half-ripe"] * L_half_ripe + percentages["ripe"] * L_ripe) / 100, 2)

    # Humidity (% RH)
    H_unripe, H_half_ripe, H_ripe = 90, 80, 72.5  # Midpoints of each range
    H_set = round((percentages["unripe"] * H_unripe + percentages["half-ripe"] * H_half_ripe + percentages["ripe"] * H_ripe) / 100, 2)

    return {
        "temperature_setpoint": f"{T_set} °C",
        "light_intensity_setpoint": f"{L_set} lux",
        "humidity_setpoint": f"{H_set} %RH"
    }

@app.route('/predict', methods=['POST'])
def predict():
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    files = request.files.getlist('files')  # Multiple file support
    
    if len(files) == 0:
        return jsonify({"error": "No files provided"}), 400

    category_detections = defaultdict(list)
    processed_images = []

    for file in files:
        image = Image.open(file).convert("RGB")
        
        img_path = f"static/predictions/{file.filename}"
        image.save(img_path)

        # YOLOv8 Predictions
        yolo_preds = yolo_model(img_path, conf=0.5)

        # Extract detected class indices and confidence scores
        yolo_classes = [int(box.cls) for box in yolo_preds[0].boxes]
        yolo_scores = [float(box.conf) for box in yolo_preds[0].boxes]

        # Save YOLOv8 processed image
        yolo_output_path = f"static/predictions/yolo_{file.filename}"
        yolo_preds[0].save(filename=yolo_output_path)
        processed_images.append(yolo_output_path)

        # Store detections per category
        for cls_idx, score in zip(yolo_classes, yolo_scores):
            cls_name = class_names[cls_idx]  # Get class name from YAML
            category = category_mapping.get(cls_name, "unknown")  # Map to unripe, half-ripe, or ripe
            category_detections[category].append(score)

    # Compute count and average confidence per category
    category_results = {
        category: {
            "count": len(scores),
            "average_confidence": round(sum(scores) / len(scores), 4) if scores else 0
        }
        for category, scores in category_detections.items()
    }

    # Compute ripeness percentages
    ripeness_percentages = calculate_percentage({k: v["count"] for k, v in category_results.items()})

    # Compute environmental recommendations
    recommendations = environmental_recommendations(ripeness_percentages)

    return jsonify({
        "predictions": category_results,
        "ripeness_percentages": ripeness_percentages,
        "environmental_recommendations": recommendations,
        "yolo_images": processed_images
    })

@app.route('/get_image/<path:filename>', methods=['GET'])
def get_image(filename):
    return send_file(f"static/predictions/{filename}", mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

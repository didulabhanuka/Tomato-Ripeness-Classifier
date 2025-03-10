import os
import yaml
from PIL import Image
from collections import defaultdict
from models.yolo_model import yolo_model

# Load Class Names from YAML
yaml_path = "config.yaml"
with open(yaml_path, "r") as f:
    config = yaml.safe_load(f)

class_names = config["names"]

# Class Mapping
category_mapping = {
    class_names[2]: "unripe",
    class_names[5]: "unripe",
    class_names[1]: "half-ripe",
    class_names[4]: "half-ripe",
    class_names[0]: "ripe",
    class_names[3]: "ripe"
}

def calculate_percentage(counts):
    """Calculate ripeness percentage."""
    total = sum(counts.values())
    return {
        "unripe": round((counts.get("unripe", 0) / total) * 100, 2) if total else 0.0,
        "half-ripe": round((counts.get("half-ripe", 0) / total) * 100, 2) if total else 0.0,
        "ripe": round((counts.get("ripe", 0) / total) * 100, 2) if total else 0.0
    }

def environmental_recommendations(percentages):
    """Calculate recommended temperature, light intensity, and humidity."""
    temp = round((percentages["unripe"] * 20 + percentages["half-ripe"] * 22 + percentages["ripe"] * 24) / 100, 2)
    light = round((percentages["unripe"] * 7000 + percentages["half-ripe"] * 5000 + percentages["ripe"] * 3000) / 100, 2)
    humidity = round((percentages["unripe"] * 90 + percentages["half-ripe"] * 80 + percentages["ripe"] * 72.5) / 100, 2)

    return {
        "temperature_setpoint": f"{temp} °C",
        "light_intensity_setpoint": f"{light} lux",
        "humidity_setpoint": f"{humidity} %RH"
    }

def process_images(files):
    """Process multiple images and return combined results."""
    category_detections = defaultdict(list)
    processed_images = []

    for file in files:
        image = Image.open(file).convert("RGB")
        img_path = f"static/predictions/{file.filename}"
        image.save(img_path)

        # Run YOLO detection
        yolo_preds = yolo_model(img_path, conf=0.5)
        yolo_classes = [int(box.cls) for box in yolo_preds[0].boxes]
        yolo_scores = [float(box.conf) for box in yolo_preds[0].boxes]

        yolo_output_path = f"static/predictions/yolo_{file.filename}"
        yolo_preds[0].save(filename=yolo_output_path)
        processed_images.append(yolo_output_path)

        for cls_idx, score in zip(yolo_classes, yolo_scores):
            cls_name = class_names[cls_idx]
            category = category_mapping.get(cls_name, "unknown")
            category_detections[category].append(score)

    # Compute final count and average confidence per category
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

    return {
        "predictions": category_results,
        "ripeness_percentages": ripeness_percentages,
        "environmental_recommendations": recommendations,
        "yolo_images": processed_images
    }

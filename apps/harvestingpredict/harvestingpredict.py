import os
import yaml
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from PIL import Image
from collections import defaultdict
from apps.harvestingpredict.models.yolo_model import yolo_model

# Firebase Initialization
cred = credentials.Certificate("harvesta-24-25j-250-firebase-adminsdk.json")  # Add your Firebase credentials
firebase_admin.initialize_app(cred)
db = firestore.client()

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

def fetch_growth_speed():
    """Fetch the latest growth speed of ripe tomatoes from Firebase."""
    try:
        growth_ref = db.collection("growth_rates").order_by("date", direction=firestore.Query.DESCENDING).limit(1).get()
        if growth_ref and growth_ref[0].exists:
            growth_speed = growth_ref[0].to_dict().get("growth_speed_ripe")
            return float(growth_speed) if growth_speed is not None else 5.0  # Default fallback to 5% per day
    except Exception as e:
        print(f"Error fetching growth speed: {e}")
    return 5.0  # Fallback default growth speed if Firebase retrieval fails


def calculate_percentage(counts):
    """Calculate ripeness percentage."""
    total = sum(counts.values())
    return {
        "unripe": round((counts.get("unripe", 0) / total) * 100, 2) if total else 0.0,
        "half-ripe": round((counts.get("half-ripe", 0) / total) * 100, 2) if total else 0.0,
        "ripe": round((counts.get("ripe", 0) / total) * 100, 2) if total else 0.0
    }

def calculate_harvest_time(ripe_percentage):
    """Calculate the optimal harvest time in days."""
    growth_speed = fetch_growth_speed()
    return round((70 - ripe_percentage) / growth_speed, 2)

def environmental_recommendations(percentages):
    """Calculate recommended environmental conditions."""
    temp = round((percentages["unripe"] * 20 + percentages["half-ripe"] * 22 + percentages["ripe"] * 24) / 100, 2)
    light = round((percentages["unripe"] * 7000 + percentages["half-ripe"] * 5000 + percentages["ripe"] * 3000) / 100, 2)
    humidity = round((percentages["unripe"] * 90 + percentages["half-ripe"] * 80 + percentages["ripe"] * 72.5) / 100, 2)

    return {
        "temperature_setpoint": f"{temp} °C",
        "light_intensity_setpoint": f"{light} lux",
        "humidity_setpoint": f"{humidity} %RH"
    }


def save_growth_speed(ripe_percentage):
    """Save the calculated growth speed to Firebase."""
    try:
        growth_speed = fetch_growth_speed()  # Fetch latest growth speed (fallback if needed)
        
        # Save new entry with timestamp
        data = {
            "date": datetime.utcnow().isoformat(),
            "ripe_percentage": ripe_percentage,
            "growth_speed_ripe": growth_speed,
            "temperature": 22,  # Example: Can be replaced with real values
            "light_intensity": 5000,
            "humidity": 80
        }
        
        db.collection("growth_rates").add(data)
        print(f"Saved growth speed to Firebase: {data}")

    except Exception as e:
        print(f"Error saving growth speed: {e}")

from datetime import datetime

def save_ripeness_data(ripeness_percentages):
    """Save all ripeness stage percentages to Firebase Firestore."""
    try:
        data = {
            "date": datetime.utcnow().isoformat(),
            "unripe_percentage": ripeness_percentages.get("unripe", 0.0),
            "half_ripe_percentage": ripeness_percentages.get("half-ripe", 0.0),
            "ripe_percentage": ripeness_percentages.get("ripe", 0.0),
        }

        # Add entry to Firestore collection "growth_rates"
        db.collection("growth_rates").add(data)
        print(f"Saved ripeness data to Firebase: {data}")

    except Exception as e:
        print(f"Error saving ripeness data: {e}")

def process_images(files):
    """Process multiple images, calculate ripeness, and save to Firebase."""
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

    # Save ripeness data in Firebase
    save_ripeness_data(ripeness_percentages)

    # Compute harvesting time
    harvest_time = calculate_harvest_time(ripeness_percentages["ripe"])

    # Compute environmental recommendations
    recommendations = environmental_recommendations(ripeness_percentages)

    return {
        "predictions": category_results,
        "ripeness_percentages": ripeness_percentages,
        "optimal_harvest_time_days": harvest_time,
        "environmental_recommendations": recommendations,
        "yolo_images": processed_images
    }

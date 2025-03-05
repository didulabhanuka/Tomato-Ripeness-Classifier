import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Define class names
class_names = ["b_fully_ripened", "b_half_ripened", "b_green", "l_fully_ripened", "l_half_ripened", "l_green"]

# Load Faster R-CNN model
frcnn_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = len(class_names) + 1
in_features = frcnn_model.roi_heads.box_predictor.cls_score.in_features
frcnn_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Force model to load on CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
frcnn_model.load_state_dict(torch.load("tomato_ripeness_classifier/fasterrcnn_tomato_model/model.pth", map_location=device))

frcnn_model.to(device)
frcnn_model.eval()

import torch
from efficientnet_pytorch import EfficientNet
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import ImageEnhance

def histogram_equalization(img):
    img = np.array(img)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img)

def median_filter(img, kernel_size=3):
    img = np.array(img)
    img = cv2.medianBlur(img, kernel_size)
    return Image.fromarray(img)

def enhance_image(img):
    img = ImageEnhance.Color(img).enhance(1.2)  # Adjust color balance
    img = ImageEnhance.Contrast(img).enhance(1.2)  # Adjust contrast
    img = ImageEnhance.Sharpness(img).enhance(1.2)  # Adjust sharpness
    return img

# Load the model
model = EfficientNet.from_pretrained('efficientnet-b0')
num_classes = 4
model._fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(model._fc.in_features, num_classes)
)
model.load_state_dict(torch.load('path_to_your_model.pth'))
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Lambda(histogram_equalization),
    transforms.Lambda(median_filter),
    transforms.Lambda(enhance_image),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load and preprocess the image
img = Image.open('path_to_your_image.jpg')
img = transform(img).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(img)
    _, predicted = torch.max(output, 1)
    severity = predicted.item()
    print(f'Predicted Acne Severity: {severity}')

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
from concurrent.futures import ThreadPoolExecutor

# Load class names from a file
def load_class_names(file_path):
    with open(file_path, 'r') as file:
        class_names = [line.strip() for line in file]
    return class_names

# Load the model with the correct architecture
def load_model():
    model = models.resnet18(weights='IMAGENET1K_V1')  # Use 'weights' instead of 'pretrained'
    model.fc = nn.Linear(model.fc.in_features, 1000)  # Adjust to the original model's output units

    # Load the saved model weights
    model.load_state_dict(torch.load('E:/PROJECTS/FINAL YEAR PROJECT/CODE/Model Dataset/fish60.pth'))
    model.eval()
    
    return model

# Define the image preprocessing
def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return preprocess(image)

# Make sure to load the correct model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model().to(device)

# Define the prediction function
def predict_species(model, image):
    input_tensor = preprocess_image(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output, dim=1)
    top3_prob, top3_classes = probabilities.topk(3, dim=1)

    class_names = load_class_names('E:/PROJECTS/FINAL YEAR PROJECT/Tracker/model/classes.txt')

    top3_probabilities = top3_prob[0].cpu().numpy()
    top3_classes = top3_classes[0].cpu().numpy()

    results = []
    for i in range(3):
        predicted_class_name = class_names[top3_classes[i]]
        results.append((predicted_class_name, top3_probabilities[i]))
    
    return results

def process_frame(frame):
    # Apply background subtraction
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 253, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    processed_frames = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            if h >= 50 and w >= 100:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                
                roi = frame[y:y + h, x:x + w]
                roi_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

                results = predict_species(model, roi_image)
                
                label = f"Species: {results[0][0]} ({results[0][1]:.2f})" if results else "Unknown"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    processed_frames.append(frame)
    return processed_frames

url = 'http://192.168.176.129:81/stream'
cap = cv2.VideoCapture(url)

object_detector = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=35)

# Use a thread pool for processing frames
executor = ThreadPoolExecutor(max_workers=20)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    future = executor.submit(process_frame, frame)
    processed_frames = future.result()

    for processed_frame in processed_frames:
        cv2.imshow("Frame", processed_frame)
    
    key = cv2.waitKey(30)
    if key == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
executor.shutdown()

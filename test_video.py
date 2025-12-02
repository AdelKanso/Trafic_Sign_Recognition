import cv2
import torch
from torchvision import transforms

from neural_network.cnn import CNN
from utils.constants import SIGN_CLASSES

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = CNN(num_classes=43).to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load your YouTube video
cap = cv2.VideoCapture("dataset/cropped.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize to (32x32)
    img = cv2.resize(frame, (32, 32))
    img = transform(img).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(1).item()
        label = SIGN_CLASSES[pred]

    # Show label on video
    cv2.putText(frame, label, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("YouTube Traffic Sign Test", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()

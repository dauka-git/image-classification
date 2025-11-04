import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

class SimpleCIFAR10Net(nn.Module):
    def __init__(self):
        super(SimpleCIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource
def load_pytorch_model(model_path='./cifar_net.pth'):
    model = SimpleCIFAR10Net()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_pytorch_model()
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

st.title("CIFAR-10 Classifier")

uploaded_file = st.file_uploader("Choose an image", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    st.subheader("Prediction Results")
    st.write(f"**Class:** {classes[predicted.item()]}")
    st.write(f"**Confidence:** {confidence.item():.2%}")
    
    st.subheader("All Predictions")
    probs = probabilities.squeeze().numpy()
    for i, class_name in enumerate(classes):
        st.write(f"{class_name}: {probs[i]:.2%}")
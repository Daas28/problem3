import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np

class DigitGenerator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.main = torch.nn.Sequential(
            torch.nn.Linear(100, 256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256, 512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 784),
            torch.nn.Tanh()
        )
    
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)

@st.cache_resource
def load_model():
    model = DigitGenerator()
    model.load_state_dict(torch.load('mnist_generator.pth', map_location='cpu'))
    model.eval()
    return model

def generate_images(model, num=5):
    images = []
    for i in range(num):
        with torch.no_grad():
            noise = torch.randn(1, 100)
            img = model(noise).squeeze().numpy()
            images.append(img)
    return images

model = load_model()
st.title('Générateur de Chiffres MNIST')

digit = st.selectbox('Choisir un chiffre (0-9)', range(10))
if st.button('Générer'):
    images = generate_images(model)
    
    cols = st.columns(5)
    for i, img in enumerate(images):
        with cols[i]:
            fig, ax = plt.subplots()
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            st.pyplot(fig)
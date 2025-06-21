import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# Debug info
st.write("Working directory:", os.getcwd())
st.write("Files in directory:", os.listdir())

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
    try:
        model = DigitGenerator()
        model.load_state_dict(torch.load('streamlit_app.py', map_location='cpu'))
        model.eval()
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def main():
    st.title("MNIST Digit Generator")
    
    model = load_model()
    if model is None:
        st.stop()
    
    digit = st.selectbox("Select digit (0-9):", range(10))
    
    if st.button("Generate"):
        with st.spinner("Generating..."):
            try:
                images = []
                for _ in range(5):
                    noise = torch.randn(1, 100)
                    with torch.no_grad():
                        img = model(noise).squeeze().numpy()
                    images.append(img)
                
                cols = st.columns(5)
                for i, img in enumerate(images):
                    with cols[i]:
                        fig, ax = plt.subplots()
                        ax.imshow(img, cmap='gray')
                        ax.axis('off')
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"Generation failed: {str(e)}")

if __name__ == "__main__":
    main()

import streamlit as st
import torch
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pytorch'))
from model import LeNet5

# Load model
device = torch.device("cpu")
model = LeNet5().to(device)
model.load_state_dict(torch.load(
    os.path.join(os.path.dirname(__file__), '..', 'pytorch', 'lenet5_mnist.pth'),
    map_location=device
))
model.eval()

st.title("🔢 LeNet-5 Digit Recognizer")
st.write("Draw a digit (0-9) below and see the model predict it in real time!")

# Drawing canvas
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data.astype(np.uint8)
    img = Image.fromarray(img).convert('L')
    img_array_full = np.array(img)

    # Find bounding box of the drawn digit
    coords = np.argwhere(img_array_full > 20)

    if coords.shape[0] > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Crop to the digit's bounding box
        cropped = img_array_full[y_min:y_max+1, x_min:x_max+1]
        cropped_img = Image.fromarray(cropped)

        # Resize so the digit fills ~20x20, leaving padding like real MNIST
        cropped_img.thumbnail((20, 20), Image.LANCZOS)

        # Paste onto a centered 28x28 black canvas
        final_img = Image.new('L', (28, 28), color=0)
        paste_x = (28 - cropped_img.width) // 2
        paste_y = (28 - cropped_img.height) // 2
        final_img.paste(cropped_img, (paste_x, paste_y))

        img_array = np.array(final_img).astype(np.float32) / 255.0

        st.write("This is what the model sees (28x28):")
        st.image(img_array, width=140, clamp=True)

        img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)

        if st.button("Predict"):
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                prediction = probs.argmax(dim=1).item()
                confidence = probs[0][prediction].item()

            st.success(f"Predicted Digit: **{prediction}** (Confidence: {confidence*100:.2f}%)")
            st.bar_chart(probs[0].numpy())
    else:
        st.write("Draw something first!")
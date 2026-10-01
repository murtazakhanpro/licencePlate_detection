import gradio as gr
from ultralytics import YOLO

# 1. Use a raw string (r"...") for Windows paths to avoid escape character errors
model = YOLO(r"D:\Computer_Vision_Projects\APDS_yolo_LARGE\YOLO_Large_ANPDS.pt")


def pred(image):
    # 2. Run prediction on the input image
    results = model.predict(image)

    # 3. Extract the first result and plot the bounding boxes
    # results[0].plot() returns a NumPy array of the image with annotations
    annotated_image = results[0].plot()

    return annotated_image


# 4. Launch the Gradio interface
app = gr.Interface(fn=pred, inputs="image", outputs="image")

if __name__ == "__main__":
    app.launch(share=True)

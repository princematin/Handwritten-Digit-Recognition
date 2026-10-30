# Imports
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import joblib



# ----------------- //// GUI Functions ////-----------------



# Variables
is_drawing = False
x, y = 0, 0
drawing_strokes = []
ready_image = None


# clear paint window
def clear_window():
    global x, y, is_drawing, drawing_strokes
    drawing_strokes = []
    canvas.delete("all")
    is_drawing = False
    x, y = 0, 0


# start drawing number
def start_drawing(event):
    global is_drawing, x, y, drawing_strokes

    is_drawing = True

    x, y = event.x, event.y

    drawing_strokes.append([])
    drawing_strokes[-1].append((x, y))

# draw function
def draw(event):
    global x, y, drawing_strokes

    if is_drawing:    
        new_x, new_y = event.x, event.y

        canvas.create_line(x, y, new_x, new_y, width=16, fill="white")

        x, y = new_x, new_y

        drawing_strokes[-1].append((new_x, new_y))


# stop drawing
def stop_drawing(event):
    global is_drawing
    is_drawing = False


# create picture function
def create_image():
    global ready_image

    image = Image.new(mode="L", size=(240, 240), color=0)
    draw = ImageDraw.Draw(image)
    for stroke in drawing_strokes:
        draw.line(xy=stroke, fill=255, width=16)
    ready_image = image.resize(size=(28, 28), resample=Image.Resampling.LANCZOS)

    return ready_image

    


# ----------------- //// Train model/ model Functions //// -----------------



# image to array
def image_to_array(picture):
    array_image = np.array(picture).astype("float32").flatten().reshape(1, 784)
    array_image /= 255.0
    return array_image


# model answer
def predict_digit():
    image = create_image()
    array_image = image_to_array(image)
    answer = trained_model.predict(array_image)

    answer_label.config(text=f"Answer: {answer[0]}")

    return answer[0]


# load "digit_model.pk1"
def load_model():
    model = joblib.load("digit_model.pkl")
    return model



# ----------------- //// Load trained model //// -----------------



# training
trained_model = load_model()



# ----------------- //// Main GUI app //// -----------------



# screen settings
window = tk.Tk()
window.title("Handwritten Digit Recognition")
window.geometry("280x300")
window.maxsize(280, 300)
window.minsize(280, 300)
window.configure(bg="black")


# paint screen
canvas = tk.Canvas(window, width=240, height=240, bg="black")
canvas.place(x=18, y=10)
canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_drawing)


# clear button (start clear_window func)
clear_button= tk.Button(window, text="Clear", height=1, width=10, fg="red", command=clear_window)
clear_button.place(x=190, y=260)



# send the picture to AI
predict_button = tk.Button(window, text="Predict", height=1, width=10, fg="green", command=predict_digit)
predict_button.place(x=10, y=260)


# Answer label
answer_label = tk.Label(window, text=f"Answer: ", fg="blue")
answer_label.place(x=110, y=260)

window.mainloop()
# mainloop
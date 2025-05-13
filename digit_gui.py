import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Cargar modelo
model = tf.keras.models.load_model("digit_model.h5")

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Reconocedor de Dígitos")

        self.canvas = tk.Canvas(master, width=280, height=280, bg="white")
        self.canvas.pack()

        self.button_predict = tk.Button(master, text="Predecir", command=self.predict)
        self.button_predict.pack()

        self.label_result = tk.Label(master, text="Dibuja un número y presiona 'Predecir'")
        self.label_result.pack()

        self.button_clear = tk.Button(master, text="Limpiar", command=self.clear_canvas)
        self.button_clear.pack()

        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.draw_digit)

    def draw_digit(self, event):
        x, y = event.x, event.y
        r = 8
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=0)

    def predict(self):
        # Redimensionar y procesar imagen
        image = self.image.resize((28, 28))
        image = ImageOps.invert(image)
        image = np.array(image).reshape(1, 28, 28, 1).astype("float32") / 255

        # Predecir
        prediction = model.predict(image)
        digit = np.argmax(prediction)
        self.label_result.config(text=f"Número predicho: {digit}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text="Dibuja un número y presiona 'Predecir'")

# Ejecutar la app
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()

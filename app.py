import gradio as gr
import tensorflow as tf

def classify_img(img):
    img = img.reshape([-1, 300, 300, 3])
    img = img / 255.0
    prediction = model.predict(img).flatten()
    print({class_names[i]: float(prediction[i]) for i in range(5)})
    return {class_names[i]: float(prediction[i]) for i in range(5)}

model = tf.keras.models.load_model('model.h5')
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic']


image = gr.inputs.Image(shape=(300, 300))
label = gr.outputs.Label(num_top_classes=5)

iface = gr.Interface(fn=classify_img, inputs=image, outputs=label)
iface.launch()
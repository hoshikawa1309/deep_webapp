import os
from flask import Flask, request, redirect, url_for, flash
# サニタイズを行う
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import numpy as np
import keras
from PIL import Image
import numpy as np
import sys


UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])
classes = ["monkey", "cat", "crow"]
num_class = len(classes)
image_size = 50


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super secret key"



def allowed_file(filename):

    '''
    .が含まれるかつ、特定の拡張子の時1を返す
    '''

    # return '.' in filename and filename.rsprit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    root, ext = os.path.splitext(filename)
    ext = ext.lower()[1:]
    return '.' in filename and ext in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # model = load_model('./animal/animal_cnn_right.h5')
            model = load_model('./animal/animal_cnn_right2.h5')
            graph = tf.compat.v1.get_default_graph()
            # global graph
            with graph.as_default():
                image = Image.open(filepath)
                image = image.convert('L')
                image = image.resize((image_size, image_size))
                data = np.asarray(image)
                data = np.reshape(data,[image_size, image_size, 1])
                X = []
                X.append(data)
                X = np.array(X)
                result = model.predict([X])[0]

                predict_label = result.argmax()
                percentage = int(result[predict_label] * 100)

                return 'ラベル : ' + classes[predict_label] + ',  確率 : ' + str(percentage) + ' %'


            # return redirect(url_for('uploaded_file', filename=filename))

    return '''
    <!doctype html>
    <html>
    <head>
    <meta charset="UTF-8">
    <title>ファイルをアップロードして判定しよう</title></head>
    <body>
    <h1>ファイルをアップロードして判定しよう！</h1>
    <form method = post enctype = multipart/form-data>
    <p><input type=file name=file>
    <input type=submit value=Upload>
    </form>
    </body>
    </html>
    '''


from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
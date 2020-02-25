from flask import Flask, request, render_template, jsonify, Response, redirect, url_for
import json
import sys
import nbformat
from nbformat.v4 import new_notebook, new_code_cell
import os
import time
import numpy as np
#from workzeug.utils import secure_filename

modelsummary = ''
testoutput = ''
UPLOAD_FOLDER = '.'
ALLOWED_EXTENSIONS = {'csv'}
train_file_type = ''
train_file_name = ''
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return Response(open('index.html').read(), mimetype='text/html')


@app.route('/fileupload', methods=['POST'])
def fileupload():
    global train_file_type
    global train_file_name
    if 'file' in request.files:
        f = request.files['file']
        train_file_name = f.filename
        f.save(f.filename)
        train_file_type = 'uploaded'
    return ('', 204)


@app.route('/process', methods=['POST'])
def process():
    global train_file_name
    global train_file_type
    f = open("demofile.py", "w")
    nb = new_notebook()
    json_string = request.form['state']
    datastore = json.loads(json_string)
    dataset = datastore['dataset']
    activations = ["Dropout", "MaxPooling"]
    optimizer = str(datastore["optimizer"])
    content = '''
import tensorflow as tf
logdir="logboard"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)'''

    if train_file_type=='uploaded':
        content += '''
from sklearn.model_selection import train_test_split

'''
    else:
         content += '''
dataset = tf.keras.datasets.{}
(x_train, y_train), (x_test, y_test) = dataset.load_data()'''.format(dataset)

    content+='''
model = tf.keras.models.Sequential([
    tf.keras.layers.{}(input_shape=({}))'''.format(datastore["layers"]["1"]["type"], datastore["layers"]["1"]["inputsize"])

    for i in range(2, datastore["layers"]["count"]+1):
        
        if datastore["layers"][str(i)]["type"] in activations:
            content = content + '''
        ,tf.keras.layers.{}({})'''.format(datastore['layers'][str(i)]['type'], 
        datastore["layers"][str(i)]["outputsize"])

        elif datastore["layers"][str(i)]["type"] == 'Conv2D':
            content = content + '''
        ,tf.keras.layers.{}({}, {}, activation=tf.nn.{})'''.format(datastore["layers"][str(i)]['type'], 
        int(datastore["layers"][str(i)]['outputsize']), 
        datastore["layers"][str(i)]['inputsize'],
        datastore["layers"][str(i)]["act"])
        
        else:
            content = content + '''
        ,tf.keras.layers.{}({}, activation=tf.nn.{})'''.format(datastore["layers"][str(i)]['type'], 
        int(datastore["layers"][str(i)]['outputsize']), 
        datastore["layers"][str(i)]["act"])

    content = content + '''])
model.compile(optimizer='{}',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])
model.evaluate(x_test,  y_test, verbose=2)
model.save('returnfile.h5')'''.format(optimizer)
    f.write(content)
    nb.cells.append(new_code_cell(content))
    nbformat.write(nb, 'demofile.ipynb')
    f.close()
    os.system('kaggle kernels push')
    time.sleep(60*2)
    os.system(
        'kaggle kernels output akashdeepsingh8888/demofile2 -p ./test/modelfile')
    return ('', 204)


@app.route('/resultpage', methods=['GET', 'POST'])
def resultpage():
    return render_template('resultpage.html', model_summary=modelsummary, testoutput=testoutput)


@app.route('/testinput', methods=['POST'])
def testinput():
    json_string = request.form['state']
    testinput = json.loads(json_string)
    testinput['input'] = testinput['input'].rstrip()
    f = open('./test/modelfile/input.csv', 'w')
    f.write(testinput['input'])
    f.close()
    os.system('kaggle datasets version -p ./test/modelfile -m "update"')
    runfilecontent = r'''
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
trained_model = tf.keras.models.load_model('../input/modelfile/returnfile.h5')
testinput = pd.read_csv('../input/modelfile/input.csv',header=None).to_numpy()
testinput = testinput.reshape((1,{}))
print(trained_model.summary())
testoutput = trained_model.predict(testinput)
with open('modelsummary.txt','w') as ms:
    trained_model.summary(print_fn=lambda x: ms.write(x+'\n'))

with open('testoutput.txt','w') as testwriterfile:
    testwriterfile.write(str(testoutput))
    '''.format(testinput['shape'])
    f = open('./test/test.py', 'w')
    f.write(runfilecontent)
    f.close()
    os.system('kaggle kernels push -p ./test')
    time.sleep(60)
    os.system('kaggle kernels output akashdeepsingh8888/testfile2 -p ./test')
    global modelsummary
    global testoutput
    with open('./test/modelsummary.txt', 'r') as f:
        modelsummary = f.read()
    with open('./test/testoutput.txt', 'r') as f:
        testoutput = f.read()
    os.system('tensorboard --logdir ./test/modelfile/logboard')
    return ('', 204)


if __name__ == "__main__":
    app.run(debug=True)

"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 15 October, 2017 @ 3:04 PM.
  Copyright © 2017. Victor. All rights reserved.
"""

# coding: utf-8

from flask import Flask, request, jsonify
from werkzeug import secure_filename

from models.features import Features
from models.funcs import upload_file
from models.network import predict
from models import config


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.secret_key = config.SECRET_KEY


@app.route('/api/<img>', method=['GET', 'POST'])
def api(img=None):
    response = {
        'status': False,
        'msg': 'Unexpected argument. No image specified',
        'data': None
    }
    
    if img:
        img = request.json('img')
        if upload_file(img, img.filename, app.config['UPLOAD_FOLDER']):
            # preprocess image
            features = Features(data_dir=config.DATASET_PATH)
            img_class = predict(img.filename)
            
            response['status'] = True
            response['msg'] = 'Upload sucessful.'
            response['data'] = {
                'img': img
                'img_class': img_class
            }
        else:
            response['msg'] = 'Could not upload image'
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)

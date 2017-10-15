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
from models.funcs import *
from models import config


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.secret_key = config.SECRET_KEY


@app.route('/api/<img>', method=['GET', 'POST'])
def api(img=None):
    response = {
        'status': False,
        'response': {
            'msg': 'No image specified.',
            'data': None
        }
    }
    
    if img:
        img = request.json('img')
        if upload_file(image, image.filename, app.config['UPLOAD_FOLDER']):
            features = Features()
            response['status'] = True
            response['response']['msg'] = ''
            response['data'][''] = {
            
            }
            return jsonify(response)
        return jsonify(response)
    
    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)

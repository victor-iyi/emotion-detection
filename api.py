"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  Created on 15 October, 2017 @ 3:04 PM.
  Copyright © 2017. Victor. All rights reserved.
"""

# coding: utf-8

from flask import Flask, request, jsonify

app = Flask(__name__)


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
        img_file = request.json('img')
        return jsonify(response)
    else:
        return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)

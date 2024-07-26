from flask import Flask, request, jsonify
import subprocess
import json
from flask_restx import Resource, Api
from CBR import CBR
from base_skintype import BST

app = Flask(__name__)
api = Api(app)

api.add_namespace(CBR, '/recommendation')
api.add_namespace(BST, '/predict')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000) #debug 필요하면 debug=True 넣어주기 (실제 서버에서는 안넣는 것이 좋음)

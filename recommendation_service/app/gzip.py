import gzip
import json

def gzip_compress(data):
    return gzip.compress(json.dumps(data).encode('utf-8'))
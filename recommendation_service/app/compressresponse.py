import io
import gzip
import json

def gzip_compress(data: dict) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode='wb') as f:
        f.write(json.dumps(data).encode('utf-8'))
    return buf.getvalue()
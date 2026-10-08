import urllib.request

assets = [
    'index.html',
    'style.css',
    'app.js',
    'model_weights.json',
    'samples/normal_sample.wav',
    'samples/aortic_stenosis_sample.wav',
    'samples/mitral_regurgitation_sample.wav',
    'samples/mitral_stenosis_sample.wav',
    'samples/mitral_valve_prolapse_sample.wav'
]

for a in assets:
    url = f"http://localhost:8088/{a}"
    try:
        res = urllib.request.urlopen(url)
        content_len = len(res.read())
        print(f"[OK]  {a:40s} -> HTTP {res.status} ({content_len:,} bytes)")
    except Exception as e:
        print(f"[ERR] {a:40s} -> ERROR: {e}")


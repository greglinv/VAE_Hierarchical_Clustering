import os
import hashlib

def load_fileset(path):
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    fingerprints = extract_fingerprints(files)
    return fingerprints

def extract_fingerprints(files):
    fingerprints = []
    for file in files:
        fingerprint = generate_fingerprint(file)
        fingerprints.append(fingerprint)
    return fingerprints

def generate_fingerprint(file):
    # Example fingerprint generation using SHA-256
    hasher = hashlib.sha256()
    with open(file, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

import datasketch

def construct_minhash_representations(fingerprints):
    minhashes = []
    for fingerprint in fingerprints:
        minhash = datasketch.MinHash()
        for element in fingerprint:
            minhash.update(element.encode('utf8'))
        minhashes.append(minhash)
    return minhashes

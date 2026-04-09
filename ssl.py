import sys, glob
try:
    import certifi
    print(certifi.where())
except Exception:
    cands = []
    for p in glob.glob(sys.prefix + "/**/*.pem", recursive=True):
        s = p.lower()
        if s.endswith("cacert.pem") or "certifi" in s or "ca-bundle" in s or "bundle" in s:
            cands.append(p)
    if cands:
        print(cands[0])

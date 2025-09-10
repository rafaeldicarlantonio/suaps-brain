# logging/model_io.py
import time

def timed(name: str, fn, *args, **kwargs):
    t0 = time.time()
    ok, out, err = True, None, None
    try:
        out = fn(*args, **kwargs)
        return (ok, out, None, int((time.time()-t0)*1000))
    except Exception as e:
        ok, err = False, e
        return (ok, None, e, int((time.time()-t0)*1000))

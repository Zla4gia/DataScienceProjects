"""Headless smoke test: run every page with Streamlit's AppTest and report exceptions.
usage (from the app folder):  python apptest.py [substring ...]"""
import sys, glob, os, time
from streamlit.testing.v1 import AppTest

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
sys.path.insert(0, here)
only = sys.argv[1:]
bad = 0
for f in sorted(glob.glob("pages/*.py")):
    if only and not any(o in f for o in only):
        continue
    t = time.time()
    try:
        at = AppTest.from_file(f, default_timeout=600).run()
        errs = [str(e.value)[:400] for e in at.exception]
        bad += bool(errs)
        print(f"{f:44s} {'OK ' if not errs else 'ERR'} {time.time()-t:5.1f}s metrics={len(at.metric)} tables={len(at.dataframe)} charts={len(at.get('arrow_line_chart')) + len(at.get('arrow_bar_chart'))} {errs[0] if errs else ''}")
    except Exception as e:
        bad += 1
        print(f"{f:44s} FAILED {str(e)[:300]}")
print("pages with problems:", bad)

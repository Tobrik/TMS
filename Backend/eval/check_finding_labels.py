"""Coverage check for frontend/src/lib/findingLabels.ts.

Builds every evidence token a diary day can contain (binary codes, base codes of
categorical evidences, every `_@_` value token the backend knows), compiles the TS
with tsc into a temp dir and asks shortFinding() for ru/kk/en. Fails on any missing,
empty or "?"-ending label. Run from Backend/:  python eval/check_finding_labels.py [--sample]
"""
import json, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ml_model import EVIDENCES_META, _feat_idx  # noqa: E402

FRONT = Path(__file__).resolve().parents[2] / "frontend"
codes = sorted({c for c in _feat_idx if c.startswith("E_")} | {c for c, m in EVIDENCES_META.items() if m["data_type"] in ("B", "C", "M")},
               key=lambda c: (int(c.split("_")[1]), c))

with tempfile.TemporaryDirectory() as tmp:
    subprocess.run(["npx", "tsc", "src/lib/findingLabels.ts", "--outDir", tmp, "--module", "commonjs",
                    "--target", "es2020", "--skipLibCheck"], cwd=FRONT, check=True)
    js = (Path(tmp) / "findingLabels.js") if (Path(tmp) / "findingLabels.js").exists() else next(Path(tmp).rglob("findingLabels.js"))
    script = ("const m=require(process.argv[1]);const codes=JSON.parse(process.argv[2]);"
              "console.log(JSON.stringify(codes.map(c=>[c,...['ru','kk','en'].map(l=>m.shortFinding(c,l))])))")
    out = subprocess.run(["node", "-e", script, str(js), json.dumps(codes)], check=True, capture_output=True, text=True).stdout
rows = json.loads(out)
bad = [r for r in rows if any(not v or v.strip().endswith("?") for v in r[1:])]
print(f"{len(rows)} codes checked, {len(bad)} problems")
for r in bad[:30]:
    print("BAD", r)
if "--sample" in sys.argv:
    want = ["E_14","E_53","E_66","E_79","E_91","E_97","E_144","E_181","E_194","E_201","E_202","E_214","E_220",
            "E_55_@_V_89","E_57_@_V_123","E_54_@_V_184","E_204_@_V_9","E_56_@_7","E_130_@_V_157","E_50"]
    d = {r[0]: r[1:] for r in rows}
    for c in want:
        print(c, *d[c], sep=" | ")
sys.exit(1 if bad else 0)

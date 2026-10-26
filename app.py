"""AquaBovine Nutrition - hydroponic fodder platform (IoT + ML + vision + ledger)."""
import base64, hashlib, io, json, random, re, time
import numpy as np, pandas as pd
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageDraw, ImageFilter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec
    KEY = ec.generate_private_key(ec.SECP256R1())
except Exception:
    KEY = None

app = Flask(__name__)

# name: (default, min, max, unit)  - every input is validated/clipped to these ranges
P = {"temperature": (22, 5, 45, "°C"), "humidity": (70, 20, 100, "%"), "ph": (6.0, 3, 9, "pH"),
     "ec": (1.8, 0.2, 4, "mS/cm"), "co2": (800, 300, 2000, "ppm"), "water_flow": (1.5, 0.1, 10, "L/min"),
     "light": (12000, 0, 40000, "lux"), "ratio": (30, 0, 100, "% legume (alfalfa)"), "green_index": (0.8, 0, 1, "0-1")}
FEATS = list(P)
SENS = ["temperature", "humidity", "ph", "ec", "co2", "water_flow", "light"]
TR = np.array([0, 20, 30, 40, 50, 60.])           # legume %, i.e. 100:0 ... 40:60
TY = [7.0, 6.8, 6.5, 6.0, 5.4, 4.8]               # yield kg/m2 (paper Table III)
TC = [13.0, 16.5, 19.2, 19.7, 20.3, 20.9]         # crude protein % (paper Table III)
PAPER = {"Random Forest": (93, 94, 96), "SVM": (88, 89, 90),
         "Gradient Boosting": (90, 91, 91), "Logistic Regression": (81, 82, 83)}


def stress(d):
    pen = (((d.temperature - 22) / 8) ** 2 + ((d.humidity - 70) / 25) ** 2 + ((d.ph - 6) / 1.2) ** 2
           + ((d.ec - 1.8) / 1.0) ** 2 + 0.5 * ((d.co2 - 800) / 600) ** 2
           + 0.7 * ((d.light - 12000) / 9000) ** 2 + 0.3 * ((d.water_flow - 1.5) / 2) ** 2)
    return np.exp(-0.15 * pen)


def make_data():
    """Synthetic tray data: 6 ratios x 3 trays x 6 cycles x 48 readings (follows the paper's design)."""
    rng = np.random.default_rng(7); parts = []
    for r in TR:
        for _ in range(18):
            n = 48
            d = pd.DataFrame({"temperature": rng.normal(22, 3, n), "humidity": rng.normal(70, 8, n),
                              "ph": rng.normal(6, .4, n), "ec": rng.normal(1.8, .35, n),
                              "co2": rng.normal(800, 150, n), "water_flow": rng.normal(1.5, .3, n),
                              "light": rng.normal(12000, 2500, n), "ratio": r, "step": np.arange(n)})
            s = stress(d)
            d["green_index"] = np.clip(.3 + .5 * s + rng.normal(0, .04, n), 0, 1)
            d["yield_kg"] = np.interp(r, TR, TY) * (.5 + .5 * s) * rng.normal(1, .02, n)
            d["cp"] = np.interp(r, TR, TC) * (.85 + .15 * s) * rng.normal(1, .015, n)
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def train():
    df = make_data(); tr = df.step < 38; X = df[FEATS]          # chronological 80/20 split per cycle
    meets = ((df.yield_kg >= 6.0) & (df.cp >= 17)).astype(int)
    ry = RandomForestRegressor(200, max_depth=12, min_samples_leaf=5, random_state=1, n_jobs=-1).fit(X[tr], df.yield_kg[tr])
    rc = RandomForestRegressor(200, max_depth=12, min_samples_leaf=5, random_state=1, n_jobs=-1).fit(X[tr], df.cp[tr])
    clf = {"Random Forest": RandomForestClassifier(200, max_depth=12, min_samples_leaf=5, random_state=1, n_jobs=-1),
           "SVM": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, probability=True, random_state=1)),
           "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, random_state=1),
           "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))}
    rows = []
    for n, m in clf.items():
        m.fit(X[tr], meets[tr]); p = m.predict(X[~tr]); pr = m.predict_proba(X[~tr])[:, 1]; yt = meets[~tr]
        try: auc = roc_auc_score(yt, pr)
        except Exception: auc = 0.0
        rows.append(dict(model=n, accuracy=round(100 * accuracy_score(yt, p), 1),
                         precision=round(100 * precision_score(yt, p, zero_division=0), 1),
                         recall=round(100 * recall_score(yt, p, zero_division=0), 1),
                         f1=round(100 * f1_score(yt, p, zero_division=0), 1), auc=round(100 * auc, 1)))
    return ry, rc, clf, rows


print("Training models on simulated tray data ...")
RY, RC, CLF, ROWS = train()
print("Ready.")


def clean(d):
    out, notes = {}, []
    for k, (dv, lo, hi, u) in P.items():
        try: v = float(d[k])
        except Exception: v = float("nan")
        if v != v:
            v = dv
            if k in d: notes.append(f"{k}: not a number, default {dv} used")
        elif not lo <= v <= hi:
            notes.append(f"{k}: {v:g} outside {lo}-{hi}, clipped"); v = min(max(v, lo), hi)
        out[k] = v
    return out, notes


def predict_df(df):
    X = df[FEATS]
    return RY.predict(X), RC.predict(X), {n: m.predict_proba(X)[:, 1] for n, m in CLF.items()}


def run(env):
    y, c, pr = predict_df(pd.DataFrame([env])[FEATS])
    return dict(yield_kg=round(float(y[0]), 2), cp=round(float(c[0]), 2),
                probs={k: round(float(v[0]), 3) for k, v in pr.items()})


# ---------- signed hash-chain ledger (paper: H = SHA-256(data || ts || sig)) ----------
def sign(m):
    if KEY: return KEY.sign(m.encode(), ec.ECDSA(hashes.SHA256())).hex()
    return hashlib.sha256(("key" + m).encode()).hexdigest()


class Ledger:
    def __init__(s): s.reset()
    def h(s, data, ts, sig, prev): return hashlib.sha256((json.dumps(data, sort_keys=True) + str(ts) + sig + prev).encode()).hexdigest()
    def reset(s): s.chain = []; s.add({"type": "genesis"})
    def add(s, data):
        prev = s.chain[-1]["hash"] if s.chain else "0" * 64
        ts = time.time(); sig = sign(json.dumps(data, sort_keys=True) + str(ts))
        s.chain.append(dict(index=len(s.chain), ts=ts, data=data, sig=sig, prev=prev, hash=s.h(data, ts, sig, prev)))
    def verify(s):
        for i, b in enumerate(s.chain):
            if s.h(b["data"], b["ts"], b["sig"], b["prev"]) != b["hash"] or (i and b["prev"] != s.chain[i - 1]["hash"]):
                return i
        return None


LEDGER = Ledger()
state = {k: float(v[0]) for k, v in P.items()}


def rnd(d): return {k: round(float(v), 2) for k, v in d.items()}


def controls(s):
    return dict(lights=s["light"] < 9000,
                ventilation=s["temperature"] > 25 or s["humidity"] > 80 or s["co2"] > 1000,
                pump=s["water_flow"] < 1.0 or not 5.5 <= s["ph"] <= 6.5)


# ---------- image analysis ----------
def uri(a):
    b = io.BytesIO(); Image.fromarray(a.astype("uint8")).save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def analyze(im):
    a = np.asarray(im.convert("RGB").resize((256, 256))).astype(float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    green = (g > r * 1.05) & (g > b * 1.05) & (g > 40)
    yellow = (r > g * .9) & (r > 100) & (b < r * .7)
    gf, yf = float(green.mean()), float(yellow.mean())
    health = gf / (gf + yf) if gf + yf > .02 else 0.0
    gray = a.mean(2)
    status = "Healthy" if health >= .85 else "Early nutrient stress" if health >= .6 else "Severe stress / deficiency"
    return dict(original=uri(a), segmented=uri(np.where(green[..., None], a, a * .2)), green_index=round(gf, 3),
                yellow_pct=round(100 * yf, 1), brightness=round(float(gray.mean()) / 255, 3),
                texture=round(float(gray[green].std()) / 255, 3) if green.any() else 0.0,
                health=round(health, 3), status=status)


def make_leaf(health):
    im = Image.new("RGB", (256, 256), (55, 40, 30)); d = ImageDraw.Draw(im)
    for _ in range(140):
        x, y, L = random.randint(0, 256), random.randint(0, 256), random.randint(25, 60)
        col = (200, 190, 60) if random.random() > health else (random.randint(30, 70), random.randint(140, 200), random.randint(30, 70))
        d.ellipse([x, y, x + L, y + L // 3], fill=col)
    return im.filter(ImageFilter.GaussianBlur(1))


# ---------- CSV ----------
AL = {"temp": "temperature", "temperature_c": "temperature", "t": "temperature", "rh": "humidity",
      "relative_humidity": "humidity", "conductivity": "ec", "ec_ms_cm": "ec", "co2_ppm": "co2",
      "flow": "water_flow", "waterflow": "water_flow", "water": "water_flow", "lux": "light",
      "light_intensity": "light", "legume": "ratio", "legume_fraction": "ratio", "legume_pct": "ratio",
      "r": "ratio", "green": "green_index", "green_pixels": "green_index"}


def num(v):
    s = str(v)
    if ":" in s:
        try: return float(s.split(":")[1])
        except ValueError: return np.nan
    return pd.to_numeric(s.replace("%", ""), errors="coerce")


# ---------- routes ----------
@app.get("/")
def index(): return render_template("index.html")


@app.get("/api/params")
def params(): return jsonify(P)


@app.get("/api/sensors")
def sensors():
    for k in SENS:
        dv, lo, hi, _ = P[k]
        state[k] = min(max(state[k] + random.gauss((dv - state[k]) * .1, (hi - lo) * .01), lo), hi)
    r = run(state)
    LEDGER.add({"type": "sensor", **rnd(state), "pred_yield": r["yield_kg"], "pred_cp": r["cp"]})
    return jsonify(reading=rnd({k: state[k] for k in SENS}), controls=controls(state), prediction=r, blocks=len(LEDGER.chain))


@app.post("/api/ingest")
def ingest():
    env, notes = clean({**state, **(request.get_json(silent=True) or {})})
    state.update(env); r = run(env)
    LEDGER.add({"type": "hardware", **rnd(env)})
    return jsonify(ok=True, prediction=r, controls=controls(env), notes=notes)


@app.post("/api/predict")
def predict_route():
    env, notes = clean(request.get_json(silent=True) or {})
    r = run(env); LEDGER.add({"type": "prediction", **rnd(env), "pred_yield": r["yield_kg"], "pred_cp": r["cp"]})
    r.update(inputs=env, notes=notes, importance={k: round(float(v), 3) for k, v in zip(FEATS, RY.feature_importances_)})
    return jsonify(r)


@app.post("/api/optimize")
def optimize():
    b = request.get_json(silent=True) or {}
    env, _ = clean(b.get("env", {}))
    try: w1 = min(max(float(b.get("w1", .5)), 0), 1)
    except Exception: w1 = .5
    y, c, _ = predict_df(pd.DataFrame([{**env, "ratio": float(r)} for r in TR])[FEATS])
    nz = lambda a: (a - a.min()) / ((a.max() - a.min()) or 1)
    s = w1 * nz(y) + (1 - w1) * nz(c); i = int(s.argmax())
    rows = [dict(blend=f"{100 - int(r)}:{int(r)}", legume=int(r), yield_kg=round(float(y[j]), 2),
                 cp=round(float(c[j]), 2), score=round(float(s[j]), 3)) for j, r in enumerate(TR)]
    LEDGER.add({"type": "optimization", "best": rows[i]["blend"], "w1": round(w1, 2)})
    return jsonify(rows=rows, best=i)


@app.get("/api/image/sample")
def image_sample():
    try: h = min(max(float(request.args.get("health", .9)), 0), 1)
    except ValueError: h = .9
    return jsonify(analyze(make_leaf(h)))


@app.post("/api/image")
def image_upload():
    try: im = Image.open(request.files["image"].stream)
    except Exception: return jsonify(error="Could not read that file as an image (use JPG or PNG)."), 400
    return jsonify(analyze(im))


@app.post("/api/upload_csv")
def upload_csv():
    try:
        raw = request.files["file"].read()
        try: df = pd.read_csv(io.BytesIO(raw))
        except UnicodeDecodeError: df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    except Exception as e:
        return jsonify(error="Could not read CSV: " + str(e)[:120]), 400
    if df.empty: return jsonify(error="The CSV has no rows."), 400
    df = df.head(5000)
    df.columns = [AL.get(c, c) for c in (re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_") for c in df.columns)]
    df = df.loc[:, ~df.columns.duplicated()]
    out, filled = pd.DataFrame(index=df.index), []
    for k, (dv, lo, hi, u) in P.items():
        if k in df:
            col = df[k].map(num) if k == "ratio" else pd.to_numeric(df[k], errors="coerce")
            if k == "ratio" and 0 < col.max() <= 1: col = col * 100      # accept fractions like 0.3
        else:
            col = pd.Series(float(dv), index=df.index); filled.append(k)
        out[k] = col.fillna(dv).clip(lo, hi)
    y, c, pr = predict_df(out)
    res = out.round(2).assign(pred_yield=y.round(2), pred_cp=c.round(2), meets_prob=pr["Random Forest"].round(3))
    grp = res.groupby("ratio")[["pred_yield", "pred_cp"]].mean().round(2).reset_index()
    summary = dict(rows=len(out), mean_yield=round(float(y.mean()), 2), mean_cp=round(float(c.mean()), 2),
                   pct_meets=round(100 * float((pr["Random Forest"] > .5).mean()), 1), filled=filled)
    LEDGER.add({"type": "csv", "rows": len(out)})
    return jsonify(summary=summary, columns=list(res.columns), rows=res.head(100).values.tolist(),
                   by_ratio=grp.head(12).values.tolist())


@app.get("/api/metrics")
def metrics(): return jsonify(measured=ROWS, paper=PAPER)


@app.get("/api/ledger")
def ledger():
    return jsonify(total=len(LEDGER.chain), blocks=[{**b, "sig": b["sig"][:24] + "…"} for b in LEDGER.chain[-10:]])


@app.get("/api/ledger/verify")
def verify():
    bad = LEDGER.verify(); return jsonify(ok=bad is None, bad_index=bad, total=len(LEDGER.chain))


@app.post("/api/ledger/tamper")
def tamper():
    LEDGER.chain[len(LEDGER.chain) // 2]["data"]["tampered"] = True; return jsonify(ok=True)


@app.post("/api/ledger/reset")
def reset(): LEDGER.reset(); return jsonify(ok=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

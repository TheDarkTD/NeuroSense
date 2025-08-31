import tkinter as tk
from tkinter import messagebox
import firebase_admin
from firebase_admin import credentials, db
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import colorsys
from scipy.ndimage import gaussian_filter

# =========================
# CONFIG
# =========================
SERVICE_ACCOUNT_JSON = "serviceAccountKey.json"
DATABASE_URL = "https://bioapp-496ae-default-rtdb.firebaseio.com/"

FOOT_MASK_RIGHT = "maskright.png"
FOOT_MASK_LEFT  = "maskleft.png"

CANVAS_W, CANVAS_H = 340, 450
SIGMA = 45
GAMMA = 0.2

SENSOR_POS_RIGHT = {
    "SR1": (0.28, 0.12),
    "SR2": (0.55, 0.15),
    "SR3": (0.62, 0.45),
    "SR4": (0.49, 0.30),
    "SR5": (0.30, 0.40),
    "SR6": (0.53, 0.59),
    "SR7": (0.51, 0.72),
    "SR8": (0.49, 0.85),
    "SR9": (0.34, 0.85),
}

# =========================
# FIREBASE
# =========================
def init_firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
        firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})

def _get(path):
    return db.reference(path).get()

def list_users():
    snap = _get("Users")
    users = []
    for uid, info in snap.items():
        nome = info.get("name") or info.get("email") or uid
        users.append((uid, nome))
    return users

def list_dates(uid):
    snap = _get(f"Users/{uid}/DATA")
    if not snap: return []
    return sorted(snap.keys())

def list_coletas(uid, date_str):
    snap = _get(f"Users/{uid}/DATA/{date_str}")
    if not snap: return []
    return sorted(snap.keys())

def last_non_null(val):
    import numpy as np
    if isinstance(val, (int, float)): return float(val)
    if isinstance(val, str):
        try: return float(val.replace(",", "."))
        except: return np.nan
    if isinstance(val, list):
        for v in reversed(val):
            r = last_non_null(v)
            if np.isfinite(r): return r
    if isinstance(val, dict):
        for k in sorted(val.keys(), reverse=True):
            r = last_non_null(val[k])
            if np.isfinite(r): return r
    return np.nan

def get_coleta(uid, date_str, coleta_id):
    data = _get(f"Users/{uid}/DATA/{date_str}/{coleta_id}")
    sr_vals = {f"SR{i}": last_non_null(data.get(f"SR{i}")) for i in range(1,10)}
    ts = f"{int(data.get('hour',0)):02d}:{int(data.get('minute',0)):02d}:{int(data.get('second',0)):02d}"
    return {
        "sr_vals": sr_vals,
        "battery": data.get("battery"),
        "timestamp": ts,
        "foot_side": str(data.get("foot", "right")).lower(),
        "coleta_id": coleta_id
    }

# =========================
# HEATMAP
# =========================
def render_heatmap(sr_vals, foot_side="right"):
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    coords = SENSOR_POS_RIGHT if foot_side.startswith("right") else {k:(1-x,y) for k,(x,y) in SENSOR_POS_RIGHT.items()}

    for i in range(1,10):
        val = sr_vals[f"SR{i}"]
        if np.isnan(val): continue
        x = int(coords[f"SR{i}"][0] * CANVAS_W)
        y = int(coords[f"SR{i}"][1] * CANVAS_H)
        canvas[y, x] = val

    heatmap = gaussian_filter(canvas, sigma=SIGMA)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    heatmap = heatmap ** GAMMA

    rgba = np.zeros((CANVAS_H, CANVAS_W, 4), dtype=np.uint8)
    for y in range(CANVAS_H):
        for x in range(CANVAS_W):
            frac = heatmap[y, x] ** 0.8
            hue = (1 - frac) * 240
            r, g, b = colorsys.hsv_to_rgb(hue/360, 1.0, 1.0)
            r, g, b = int(r*255), int(g*255), int(b*255)
            if frac < 0.2:
                blend = frac / 0.2
                r = int((1-blend)*0   + blend*r)
                g = int((1-blend)*0   + blend*g)
                b = int((1-blend)*180 + blend*b)
            alpha = int(180 + 75*frac)
            rgba[y, x] = [r, g, b, alpha]

    return rgba

def show_heatmap(payload):
    heatmap = render_heatmap(payload["sr_vals"], payload["foot_side"])
    mask_path = FOOT_MASK_RIGHT if payload["foot_side"].startswith("right") else FOOT_MASK_LEFT
    foot_mask = Image.open(mask_path).convert("RGBA").resize((CANVAS_W, CANVAS_H))
    from PIL import Image as PILImage
    img_heatmap = PILImage.fromarray(heatmap, mode="RGBA")
    combined = PILImage.alpha_composite(img_heatmap, foot_mask)
    plt.figure(figsize=(4,6))
    plt.imshow(combined)
    plt.axis("off")
    plt.title(f"{payload['timestamp']} | Bateria: {payload['battery']}%")
    plt.show()

# =========================
# TKINTER APP
# =========================
def main_gui():
    init_firebase()

    root = tk.Tk()
    root.title("Visualizador de Coletas")

    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    lista_usuarios = tk.Listbox(frame, height=8, width=30)
    lista_datas = tk.Listbox(frame, height=8, width=20)
    lista_coletas = tk.Listbox(frame, height=8, width=20)

    lista_usuarios.grid(row=0, column=0, padx=5)
    lista_datas.grid(row=0, column=1, padx=5)
    lista_coletas.grid(row=0, column=2, padx=5)

    # Carrega usuários
    users = list_users()
    for uid, nome in users:
        lista_usuarios.insert(tk.END, f"{nome} ({uid})")

    def on_user_select(event):
        if not lista_usuarios.curselection(): return
        selecionado = lista_usuarios.get(lista_usuarios.curselection())
        uid = selecionado.split("(")[-1].strip(")")
        lista_datas.delete(0, tk.END)
        datas = list_dates(uid)
        for d in datas:
            lista_datas.insert(tk.END, d)
        lista_datas.uid = uid

    def on_data_select(event):
        if not lista_datas.curselection(): return
        uid = lista_datas.uid
        selecionado = lista_datas.get(lista_datas.curselection())
        lista_coletas.delete(0, tk.END)
        coletas = list_coletas(uid, selecionado)
        for c in coletas:
            lista_coletas.insert(tk.END, c)
        lista_coletas.uid = uid
        lista_coletas.date = selecionado

    def on_coleta_select(event):
        if not lista_coletas.curselection(): return
        uid = lista_coletas.uid
        date_str = lista_coletas.date
        coleta_id = lista_coletas.get(lista_coletas.curselection())
        payload = get_coleta(uid, date_str, coleta_id)
        show_heatmap(payload)

    lista_usuarios.bind("<<ListboxSelect>>", on_user_select)
    lista_datas.bind("<<ListboxSelect>>", on_data_select)
    lista_coletas.bind("<<ListboxSelect>>", on_coleta_select)

    root.mainloop()

if __name__ == "__main__":
    main_gui()

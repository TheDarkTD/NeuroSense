import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox,
    QMessageBox, QStackedWidget
)
from PySide6.QtGui import QPixmap, QImage, QIcon, QPainter, QPen, QFont, QColor
from PySide6.QtCore import Qt, QTimer
import firebase_admin
from firebase_admin import credentials, db
from PIL import Image
from scipy.ndimage import gaussian_filter
import colorsys

# =========================
# CONFIG / CALIBRAÇÃO
# =========================
SERVICE_ACCOUNT_JSON = "serviceAccountKey.json"
DATABASE_URL = "https://bioapp-496ae-default-rtdb.firebaseio.com/"
FOOT_MASK_RIGHT = "maskright.png"
FOOT_MASK_LEFT  = "maskleft.png"
LOGO_IMG = "feet.png"

CANVAS_W, CANVAS_H = 340, 450
SIGMA, GAMMA = 60, 0.7

# Calibração: 1000 ADC ≈ 67.6 kPa (ponto medido)
KPA_PER_UNIT = 67.6 / 1000.0          # ≈ 0.0676 kPa por ADC
LEGEND_MAX_KPA = 280.0                # topo da escala/legenda (ajustável)
LEGEND_TICKS_KPA = [0, 70, 140, 210, 280]  # marcas da legenda

# Posições normalizadas (pé direito). Pé esquerdo é espelhado em X.
SENSOR_POS_RIGHT = {
    "SR1": (0.28, 0.12), "SR2": (0.55, 0.15), "SR3": (0.62, 0.45),
    "SR4": (0.49, 0.30), "SR5": (0.30, 0.40), "SR6": (0.53, 0.59),
    "SR7": (0.51, 0.72), "SR8": (0.49, 0.85), "SR9": (0.34, 0.85),
}

# Nomes das regiões (ajuste livre se quiser)
SR_NAMES = {
    "SR1": "Hálux (dedão)",
    "SR2": "Metatarso medial",
    "SR3": "Metatarso lateral",
    "SR4": "Arco medial",
    "SR5": "Arco lateral",
    "SR6": "Mediopé",
    "SR7": "Calcanhar medial",
    "SR8": "Calcanhar central",
    "SR9": "Calcanhar lateral",
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
    snap = _get("Users") or {}
    return [(uid, (info or {}).get("name") or (info or {}).get("email") or uid) for uid, info in snap.items()]

def list_dates(uid, side="R"):
    node = "DATA" if side == "R" else "DATA2"
    snap = _get(f"Users/{uid}/{node}") or {}
    return sorted(snap.keys())

def list_coletas(uid, date_str, side="R"):
    node = "DATA" if side == "R" else "DATA2"
    snap = _get(f"Users/{uid}/{node}/{date_str}") or {}
    return sorted(snap.keys())

def last_non_null(val):
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

def is_valid_coleta(uid, date_str, coleta_id, side="R"):
    node = "DATA" if side == "R" else "DATA2"
    data = _get(f"Users/{uid}/{node}/{date_str}/{coleta_id}")
    if not data: return False
    return all(not np.isnan(last_non_null(data.get(f"SR{i}"))) for i in range(1, 10))

def list_coletas_validas(uid, date_str, side="R"):
    return [cid for cid in list_coletas(uid, date_str, side) if is_valid_coleta(uid, date_str, cid, side)]

def get_coleta(uid, date_str, coleta_id, side="R"):
    node = "DATA" if side == "R" else "DATA2"
    data = _get(f"Users/{uid}/{node}/{date_str}/{coleta_id}")
    if not data: return None
    sr_vals = {f"SR{i}": last_non_null(data.get(f"SR{i}")) for i in range(1,10)}
    ts = f"{int(data.get('hour',0)):02d}:{int(data.get('minute',0)):02d}:{int(data.get('second',0)):02d}"
    return {"sr_vals": sr_vals, "timestamp": ts, "foot_side": "right" if side=="R" else "left", "coleta_id": coleta_id}

# expande TODAS as amostras (frames) de uma coleta (para Movimento)
def expand_coleta(uid, date_str, coleta_id, side="R"):
    node = "DATA" if side == "R" else "DATA2"
    data = _get(f"Users/{uid}/{node}/{date_str}/{coleta_id}")
    if not data: return []
    n = max(len(data.get(f"SR{i}", [])) for i in range(1,10)) if data else 0
    frames = []
    for idx in range(n):
        sr_vals = {}
        skip = False
        for i in range(1,10):
            serie = data.get(f"SR{i}", [])
            if idx >= len(serie) or serie[idx] is None:
                skip = True; break
            try:
                val = float(str(serie[idx]).replace(",", "."))
            except:
                skip = True; break
            sr_vals[f"SR{i}"] = val
        if not skip:
            ts = f"{int(data.get('hour',0)):02d}:{int(data.get('minute',0)):02d}:{int(data.get('second',0)):02d}"
            frames.append({"sr_vals": sr_vals, "timestamp": ts, "foot_side": "right" if side=="R" else "left", "coleta_id": coleta_id})
    return frames

# =========================
# HEATMAP (kPa ABSOLUTO)
# =========================
def render_heatmap(sr_vals, foot_side="right"):
    """
    Renderiza heatmap com escala ABSOLUTA em kPa.
    Cores (azul->vermelho) correspondem a 0 .. LEGEND_MAX_KPA.
    """
    # grade em ADC (0..4095)
    canvas_adc = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)
    coords = SENSOR_POS_RIGHT if foot_side == "right" else {k: (1-x, y) for k, (x, y) in SENSOR_POS_RIGHT.items()}

    for i in range(1, 10):
        v = sr_vals.get(f"SR{i}")
        if np.isnan(v): 
            continue
        x = int(coords[f"SR{i}"][0] * CANVAS_W)
        y = int(coords[f"SR{i}"][1] * CANVAS_H)
        canvas_adc[y, x] = v  # valor cru

    # suaviza (ADC) e converte para kPa ABSOLUTO
    heat_adc = gaussian_filter(canvas_adc, sigma=SIGMA)
    heat_kpa = heat_adc * KPA_PER_UNIT

    # normaliza pela faixa ABSOLUTA (0..LEGEND_MAX_KPA) para colorização
    norm = np.clip(heat_kpa / max(LEGEND_MAX_KPA, 1e-6), 0.0, 1.0)
    frac = norm ** GAMMA

    # mapeamento HSV (240->0)
    hue = (1.0 - frac) * 240.0
    h = hue / 360.0
    s = np.ones_like(hue)
    v = np.ones_like(hue)

    i = (h * 6.0).astype(int) % 6
    f = (h * 6.0) - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])

    r = (r * 255).astype(np.uint8)
    g = (g * 255).astype(np.uint8)
    b = (b * 255).astype(np.uint8)

    # ---- Alpha proporcional à intensidade absoluta ----
    # frac é 0..1 (kPa normalizado pela legenda). Usamos um leve expoente para
    # dar contraste e zeramos opacidade quando for muito baixo.
    frac_alpha = np.clip(frac**1.2, 0.0, 1.0)
    alpha = (255 * frac_alpha).astype(np.uint8)

    # Remove totalmente ruído (não “azula” o pé inteiro)
    alpha[frac < 0.02] = 0

    return np.dstack([r, g, b, alpha])

# =========================
# AUX: pico de pressão
# =========================
def max_pressure_info(payload):
    sr_vals = payload["sr_vals"]
    max_sr = None
    max_adc = -1
    for i in range(1, 10):
        key = f"SR{i}"
        v = sr_vals.get(key, np.nan)
        if not np.isnan(v) and v > max_adc:
            max_adc = v
            max_sr = key
    if max_sr is None:
        return None
    kpa = max_adc * KPA_PER_UNIT
    nome = SR_NAMES.get(max_sr, max_sr)
    return (max_sr, nome, kpa, payload["foot_side"])

def format_peak_lines(peaks):
    if not peaks:
        return "Maior pressão: —"
    lines = []
    for sensor_id, nome, kpa, side in peaks:
        lado = "Direito" if side == "right" else "Esquerdo"
        lines.append(f"{lado}: {nome} ({sensor_id}) — {kpa:.1f} kPa")
    return "Maior pressão:\n" + "\n".join(lines)

# =========================
# BASE: LEGENDA kPa + TELA CHEIA + QUADRO INFO
# =========================
class TelaBase(QWidget):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack
        self.layout = QHBoxLayout(self)

        # Esquerda (controles)
        self.left = QVBoxLayout()
        btn_voltar = QPushButton("⬅ Voltar")
        btn_voltar.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.left.addWidget(btn_voltar)

        self.fullscreen_btn = QPushButton("⛶ Tela Cheia")
        self.fullscreen_btn.setCheckable(True)
        self.fullscreen_btn.clicked.connect(self.toggle_fullscreen)
        self.left.addWidget(self.fullscreen_btn)

        # Quadro informativo do pico de pressão
        self.peak_box = QLabel("Maior pressão: —")
        self.peak_box.setWordWrap(True)
        self.peak_box.setStyleSheet("""
            QLabel {
                background-color: rgba(255,255,255,0.10);
                border: 1px solid rgba(255,255,255,0.25);
                border-radius: 8px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        self.left.addWidget(self.peak_box)

        self.left_container = QWidget()
        self.left_container.setLayout(self.left)
        self.layout.addWidget(self.left_container, 1)

        # Direita (mapas + legenda)
        self.img_vbox = QVBoxLayout()
        self.img_layout = QHBoxLayout()
        self.label_img_left = QLabel("");  self.label_img_left.setAlignment(Qt.AlignCenter)
        self.label_img_right = QLabel(""); self.label_img_right.setAlignment(Qt.AlignCenter)
        self.img_layout.addWidget(self.label_img_left)
        self.img_layout.addWidget(self.label_img_right)
        self.img_vbox.addLayout(self.img_layout)

        # Legenda com gradiente + marcas kPa (versão legível)
        self.legend = QLabel()
        self.legend.setPixmap(self._create_legend_kpa(420, 60))
        self.img_vbox.addWidget(self.legend, alignment=Qt.AlignCenter)

        self.layout.addLayout(self.img_vbox, 3)

    def toggle_fullscreen(self):
        if self.fullscreen_btn.isChecked():
            self.left_container.hide()
            self.fullscreen_btn.setText("⛶ Sair Tela Cheia")
        else:
            self.left_container.show()
            self.fullscreen_btn.setText("⛶ Tela Cheia")

    def _create_legend_kpa(self, w=420, h=60):
        """
        Barra de legenda (azul->vermelho) com marcas em kPa.
        Margens laterais, fonte maior e fundo translúcido nos rótulos.
        """
        margin = 30            # margem lateral p/ evitar corte
        grad_h = 20            # altura da barra colorida
        pad_top = 10
        img = np.zeros((h, w, 3), dtype=np.uint8)

        # gradiente
        for x in range(margin, w - margin):
            frac = (x - margin) / max(1, (w - 2*margin - 1))
            hue = (1 - frac) * 240
            r, g, b = colorsys.hsv_to_rgb(hue/360, 1, 1)
            img[pad_top:pad_top+grad_h, x, :] = [int(r*255), int(g*255), int(b*255)]

        qimg = QImage(img.data, w, h, 3*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)

        # ticks + texto
        painter = QPainter(pix)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing)
        pen = QPen(Qt.white); pen.setWidth(2)
        painter.setPen(pen)
        font = QFont(); font.setPointSize(11); font.setBold(True)
        painter.setFont(font)

        base_y = pad_top + grad_h
        for kpa in LEGEND_TICKS_KPA:
            frac = np.clip(kpa / LEGEND_MAX_KPA, 0, 1)
            x = int(margin + frac * (w - 2*margin - 1))
            painter.drawLine(x, base_y, x, base_y + 6)
            txt = f"{int(kpa)} kPa"
            tw = painter.fontMetrics().horizontalAdvance(txt)
            th = painter.fontMetrics().height()
            painter.fillRect(x - tw//2 - 3, h - th - 4, tw + 6, th, QColor(0, 0, 0, 160))
            painter.drawText(x - tw//2, h - 6, txt)

        painter.end()
        return pix

    def _draw_foot(self, payload):
        heatmap = render_heatmap(payload["sr_vals"], payload["foot_side"])
        mask_path = FOOT_MASK_RIGHT if payload["foot_side"]=="right" else FOOT_MASK_LEFT
        mask = Image.open(mask_path).convert("RGBA").resize((CANVAS_W,CANVAS_H))
        img_heatmap = Image.fromarray(heatmap,mode="RGBA")
        combined = Image.alpha_composite(img_heatmap, mask)
        qimg = QImage(combined.tobytes(), combined.width, combined.height, QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg).scaled(520,680,Qt.KeepAspectRatio, Qt.SmoothTransformation)
        if payload["foot_side"]=="right": self.label_img_right.setPixmap(pix)
        else: self.label_img_left.setPixmap(pix)

    def set_peak_info(self, peaks_list):
        self.peak_box.setText(format_peak_lines(peaks_list))

# =========================
# MENU
# =========================
class MenuInicial(QWidget):
    def __init__(self, stack):
        super().__init__()
        layout = QVBoxLayout()

        img = QLabel()
        pix = QPixmap(LOGO_IMG).scaled(150,150,Qt.KeepAspectRatio, Qt.SmoothTransformation)
        img.setPixmap(pix)
        layout.addWidget(img, alignment=Qt.AlignCenter)

        titulo = QLabel("NeuroSense")
        titulo.setStyleSheet("font-size: 36px; font-weight: bold; color: white;")
        layout.addWidget(titulo, alignment=Qt.AlignCenter)

        btn_estatico = QPushButton("Estático")
        btn_movimento = QPushButton("Movimento")
        for b in (btn_estatico, btn_movimento):
            b.setStyleSheet("background-color:#2980b9; color:white; font-size:18px; padding:10px; border-radius:8px;")
            layout.addWidget(b, alignment=Qt.AlignCenter)

        btn_estatico.clicked.connect(lambda: stack.setCurrentIndex(1))
        btn_movimento.clicked.connect(lambda: stack.setCurrentIndex(2))
        self.setLayout(layout)

# =========================
# TELA ESTÁTICO
# =========================
class TelaEstatico(TelaBase):
    def __init__(self, stack):
        super().__init__(stack); init_firebase()
        self.combo_user = QComboBox(); self.combo_date = QComboBox()
        btn_open = QPushButton("Abrir Heatmap")
        self.left.addWidget(QLabel("Usuário:")); self.left.addWidget(self.combo_user)
        self.left.addWidget(QLabel("Data:")); self.left.addWidget(self.combo_date); self.left.addWidget(btn_open)
        self.combo_view = QComboBox(); self.combo_view.addItems(["Direito","Esquerdo","Ambos"]); self.combo_view.hide()
        self.left.addWidget(QLabel("Visualização:")); self.left.addWidget(self.combo_view)
        nav = QHBoxLayout(); self.btn_prev = QPushButton("⬅ Anterior"); self.btn_next = QPushButton("Próxima ➡")
        nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next); self.left.addLayout(nav)
        self.status = QLabel("Status: pronto"); self.left.addWidget(self.status)
        self.users = list_users()
        for uid, nome in self.users: self.combo_user.addItem(nome, uid)
        self.combo_user.currentIndexChanged.connect(self.load_dates); btn_open.clicked.connect(self.open_heatmap)
        self.btn_prev.clicked.connect(self.prev_coleta); self.btn_next.clicked.connect(self.next_coleta)
        self.idx=0; self.uid=None; self.date=None; self.coletasR=[]; self.coletasL=[]; self.show_left=False; self.show_right=False

    def load_dates(self):
        self.combo_date.clear(); uid = self.combo_user.currentData()
        if not uid: return
        for d in list_dates(uid,"R"): self.combo_date.addItem(d, d)

    def open_heatmap(self):
        self.uid=self.combo_user.currentData(); self.date=self.combo_date.currentData()
        if not self.uid or not self.date: QMessageBox.warning(self,"Erro","Selecione usuário e data"); return
        info=_get(f"Users/{self.uid}") or {}
        self.show_left=str(info.get("InsolesL","false")).lower()=="true"; self.show_right=str(info.get("InsolesR","false")).lower()=="true"
        self.coletasR=list_coletas_validas(self.uid,self.date,"R") if self.show_right else []
        self.coletasL=list_coletas_validas(self.uid,self.date,"L") if self.show_left else []
        if not self.coletasR and not self.coletasL: QMessageBox.warning(self,"Erro","Nenhuma coleta válida encontrada"); return
        self.combo_view.setVisible(self.show_left and self.show_right); self.idx=0; self.show_current()

    def show_current(self):
        self.label_img_left.clear(); self.label_img_right.clear()
        peaks = []

        view_mode=self.combo_view.currentText() if self.combo_view.isVisible() else ("Direito" if self.show_right else "Esquerdo")
        max_len=max(len(self.coletasR),len(self.coletasL))
        if max_len==0:
            self.status.setText(f"Data {self.date} | Sem coletas válidas")
            self.set_peak_info([])
            return

        info_text=f"Data {self.date} | Coleta {self.idx+1}/{max_len}"
        if view_mode in ("Direito","Ambos") and self.show_right and self.idx<len(self.coletasR):
            payloadR=get_coleta(self.uid,self.date,self.coletasR[self.idx],"R")
            if payloadR:
                self._draw_foot(payloadR)
                info_text+=f" | Hora: {payloadR['timestamp']}"
                mpR = max_pressure_info(payloadR)
                if mpR: peaks.append(mpR)

        if view_mode in ("Esquerdo","Ambos") and self.show_left and self.idx<len(self.coletasL):
            payloadL=get_coleta(self.uid,self.date,self.coletasL[self.idx],"L")
            if payloadL:
                self._draw_foot(payloadL)
                info_text+=f" | Hora: {payloadL['timestamp']}"
                mpL = max_pressure_info(payloadL)
                if mpL: peaks.append(mpL)

        self.status.setText(info_text)
        self.set_peak_info(peaks)

    def next_coleta(self):
        max_len=max(len(self.coletasR),len(self.coletasL))
        if max_len==0: return
        self.idx=(self.idx+1)%max_len; self.show_current()

    def prev_coleta(self):
        max_len=max(len(self.coletasR),len(self.coletasL))
        if max_len==0: return
        self.idx=(self.idx-1)%max_len; self.show_current()

# =========================
# TELA MOVIMENTO (todas as amostras)
# =========================
class TelaMovimento(TelaBase):
    def __init__(self, stack):
        super().__init__(stack); init_firebase()
        self.combo_user=QComboBox(); self.combo_date=QComboBox(); btn_open=QPushButton("Abrir Heatmap")
        self.left.addWidget(QLabel("Usuário:")); self.left.addWidget(self.combo_user)
        self.left.addWidget(QLabel("Data:")); self.left.addWidget(self.combo_date); self.left.addWidget(btn_open)
        self.combo_view=QComboBox(); self.combo_view.addItems(["Direito","Esquerdo","Ambos"]); self.combo_view.hide()
        self.left.addWidget(QLabel("Visualização:")); self.left.addWidget(self.combo_view)
        nav=QHBoxLayout(); self.btn_prev=QPushButton("⬅ Anterior"); self.btn_next=QPushButton("Próxima ➡")
        nav.addWidget(self.btn_prev); nav.addWidget(self.btn_next); self.left.addLayout(nav)
        auto=QHBoxLayout(); self.btn_play=QPushButton("▶ Play"); self.btn_pause=QPushButton("⏸ Pause"); self.btn_stop=QPushButton("⏹ Stop")
        auto.addWidget(self.btn_play); auto.addWidget(self.btn_pause); auto.addWidget(self.btn_stop); self.left.addLayout(auto)
        self.speed_combo=QComboBox(); self.speed_combo.addItems(["Lento (1 fps)","Normal (5 fps)","Rápido (10 fps)"])
        self.left.addWidget(QLabel("Velocidade:")); self.left.addWidget(self.speed_combo)
        self.status=QLabel("Status: pronto"); self.left.addWidget(self.status)

        self.users=list_users()
        for uid, nome in self.users:
            self.combo_user.addItem(nome, uid)
        self.combo_user.currentIndexChanged.connect(self.load_dates)
        btn_open.clicked.connect(self.open_heatmap)
        self.btn_prev.clicked.connect(self.prev_frame); self.btn_next.clicked.connect(self.next_frame)
        self.btn_play.clicked.connect(self.play); self.btn_pause.clicked.connect(self.pause); self.btn_stop.clicked.connect(self.stop)

        self.idx=0; self.uid=None; self.date=None
        self.framesR=[]; self.framesL=[]
        self.show_left=False; self.show_right=False

        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)

    def load_dates(self):
        self.combo_date.clear(); uid = self.combo_user.currentData()
        if not uid: return
        for d in list_dates(uid,"R"): self.combo_date.addItem(d, d)

    def open_heatmap(self):
        self.uid = self.combo_user.currentData(); self.date = self.combo_date.currentData()
        if not self.uid or not self.date:
            QMessageBox.warning(self,"Erro","Selecione usuário e data"); return
        info = _get(f"Users/{self.uid}") or {}
        self.show_left  = str(info.get("InsolesL","false")).lower()=="true"
        self.show_right = str(info.get("InsolesR","false")).lower()=="true"

        # carrega TODAS as amostras (frames)
        self.framesR=[]; self.framesL=[]
        if self.show_right:
            for cid in list_coletas(self.uid, self.date, "R"):
                self.framesR.extend(expand_coleta(self.uid, self.date, cid, "R"))
        if self.show_left:
            for cid in list_coletas(self.uid, self.date, "L"):
                self.framesL.extend(expand_coleta(self.uid, self.date, cid, "L"))

        if not self.framesR and not self.framesL:
            QMessageBox.warning(self,"Erro","Nenhuma leitura válida encontrada"); return

        self.combo_view.setVisible(self.show_left and self.show_right)
        self.idx=0; self.show_current()

    def show_current(self):
        self.label_img_left.clear(); self.label_img_right.clear()
        peaks = []

        view_mode = self.combo_view.currentText() if self.combo_view.isVisible() else ("Direito" if self.show_right else "Esquerdo")

        max_len = max(len(self.framesR), len(self.framesL))
        if max_len == 0:
            self.status.setText(f"Data {self.date} | Sem leituras válidas")
            self.set_peak_info([])
            return

        info_text = f"Data {self.date} | Frame {self.idx+1}/{max_len}"

        if view_mode in ("Direito","Ambos") and self.show_right and self.idx < len(self.framesR):
            payloadR = self.framesR[self.idx]
            self._draw_foot(payloadR)
            info_text += f" | Hora: {payloadR['timestamp']}"
            mpR = max_pressure_info(payloadR)
            if mpR: peaks.append(mpR)

        if view_mode in ("Esquerdo","Ambos") and self.show_left and self.idx < len(self.framesL):
            payloadL = self.framesL[self.idx]
            self._draw_foot(payloadL)
            info_text += f" | Hora: {payloadL['timestamp']}"
            mpL = max_pressure_info(payloadL)
            if mpL: peaks.append(mpL)

        self.status.setText(info_text)
        self.set_peak_info(peaks)

    def play(self):
        choice = self.speed_combo.currentIndex()
        interval = 1000 if choice == 0 else 200 if choice == 1 else 100
        self.timer.start(interval)

    def pause(self):
        self.timer.stop()

    def stop(self):
        self.timer.stop()
        self.idx = 0
        self.show_current()

    def next_frame(self):
        max_len = max(len(self.framesR), len(self.framesL))
        if max_len == 0: return
        self.idx = (self.idx + 1) % max_len
        self.show_current()

    def prev_frame(self):
        max_len = max(len(self.framesR), len(self.framesL))
        if max_len == 0: return
        self.idx = (self.idx - 1) % max_len
        self.show_current()

# =========================
# MAIN / ESTILO GLOBAL
# =========================
class MainWindow(QStackedWidget):
    def __init__(self):
        super().__init__()
        init_firebase()

        self.menu = MenuInicial(self)
        self.estatico = TelaEstatico(self)
        self.movimento = TelaMovimento(self)

        self.addWidget(self.menu)       # 0
        self.addWidget(self.estatico)   # 1
        self.addWidget(self.movimento)  # 2
        self.setCurrentIndex(0)

        # Estilo global (degradê + botões + labels)
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                    stop:0 #0a2342, stop:1 #1e3c72);
                color: white;
            }
            QPushButton {
                background-color: #2980b9;
                color: white;
                font-size: 14px;
                padding: 8px;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #3498db;
            }
            QLabel {
                font-size: 14px;
            }
        """)

        self.setWindowIcon(QIcon(LOGO_IMG))
        self.setWindowTitle("NeuroSense")
        self.resize(1300, 780)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

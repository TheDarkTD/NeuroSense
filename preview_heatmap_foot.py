import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QMessageBox
)
from PySide6.QtGui import QPixmap, QImage, QPalette, QColor
from PySide6.QtCore import Qt
import firebase_admin
from firebase_admin import credentials, db
from PIL import Image
from scipy.ndimage import gaussian_filter

# =========================
# CONFIGURAÇÕES GERAIS
# =========================
SERVICE_ACCOUNT_JSON = "serviceAccountKey.json"   # chave de serviço Firebase (JSON baixado do console Firebase)
DATABASE_URL = "https://bioapp-496ae-default-rtdb.firebaseio.com/"  # URL do Realtime Database

# Máscaras gráficas (imagens do pé esquerdo/direito para sobreposição)
FOOT_MASK_RIGHT = "maskright.png"
FOOT_MASK_LEFT  = "maskleft.png"

# Tamanho da área de desenho (em pixels)
CANVAS_W, CANVAS_H = 340, 450

# Parâmetros do filtro de calor
SIGMA, GAMMA = 60, 0.7   # SIGMA controla suavização, GAMMA controla intensidade

# Posições dos sensores para o pé direito (normalizado entre 0–1)
# Para o pé esquerdo, o código espelha automaticamente
SENSOR_POS_RIGHT = {
    "SR1": (0.28, 0.12), "SR2": (0.55, 0.15), "SR3": (0.62, 0.45),
    "SR4": (0.49, 0.30), "SR5": (0.30, 0.40), "SR6": (0.53, 0.59),
    "SR7": (0.51, 0.72), "SR8": (0.49, 0.85), "SR9": (0.34, 0.85),
}

# =========================
# FUNÇÕES FIREBASE
# =========================
def init_firebase():
    """Inicializa a conexão com o Firebase, apenas uma vez."""
    if not firebase_admin._apps:
        cred = credentials.Certificate(SERVICE_ACCOUNT_JSON)
        firebase_admin.initialize_app(cred, {"databaseURL": DATABASE_URL})

def _get(path):
    """Lê dados de um caminho do Realtime Database."""
    return db.reference(path).get()

def list_users():
    """Lista todos os usuários no banco, retorna [(uid, nome), ...]."""
    snap = _get("Users")
    users = []
    for uid, info in snap.items():
        nome = info.get("name") or info.get("email") or uid
        users.append((uid, nome))
    return users

def list_dates(uid, side="R"):
    """Lista todas as datas disponíveis para um usuário.
       side="R" → pé direito (DATA)
       side="L" → pé esquerdo (DATA2)
    """
    node = "DATA" if side == "R" else "DATA2"
    snap = _get(f"Users/{uid}/{node}")
    return sorted(snap.keys()) if snap else []

def list_coletas(uid, date_str, side="R"):
    """Lista todos os IDs de coletas para uma data específica."""
    node = "DATA" if side == "R" else "DATA2"
    snap = _get(f"Users/{uid}/{node}/{date_str}")
    return sorted(snap.keys()) if snap else []

def last_non_null(val):
    """Retorna o último valor não-nulo de uma lista/dict de leituras."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        try:
            return float(val.replace(",", "."))
        except:
            return np.nan
    if isinstance(val, list):  # percorre de trás para frente
        for v in reversed(val):
            r = last_non_null(v)
            if np.isfinite(r):
                return r
    if isinstance(val, dict):  # percorre chaves em ordem reversa
        for k in sorted(val.keys(), reverse=True):
            r = last_non_null(val[k])
            if np.isfinite(r):
                return r
    return np.nan

def is_valid_coleta(uid, date_str, coleta_id, side="R"):
    """Verifica se uma coleta é válida (todos os sensores possuem dados não-nulos)."""
    node = "DATA" if side == "R" else "DATA2"
    data = _get(f"Users/{uid}/{node}/{date_str}/{coleta_id}")
    if not data:
        return False
    for i in range(1, 10):
        v = last_non_null(data.get(f"SR{i}"))
        if np.isnan(v):
            return False
    return True

def list_coletas_validas(uid, date_str, side="R"):
    """Lista apenas as coletas válidas (sem sensores nulos)."""
    return [cid for cid in list_coletas(uid, date_str, side) if is_valid_coleta(uid, date_str, cid, side)]

def get_coleta(uid, date_str, coleta_id, side="R"):
    """Carrega os dados de uma coleta específica (sensores + timestamp)."""
    node = "DATA" if side == "R" else "DATA2"
    data = _get(f"Users/{uid}/{node}/{date_str}/{coleta_id}")
    if not data:
        return None
    sr_vals = {f"SR{i}": last_non_null(data.get(f"SR{i}")) for i in range(1, 10)}
    ts = f"{int(data.get('hour',0)):02d}:{int(data.get('minute',0)):02d}:{int(data.get('second',0)):02d}"
    return {
        "sr_vals": sr_vals,
        "timestamp": ts,
        "foot_side": "right" if side == "R" else "left",
        "coleta_id": coleta_id
    }

# =========================
# HEATMAP (MAPA DE CALOR)
# =========================
def render_heatmap(sr_vals, foot_side="right"):
    """Gera o mapa de calor como imagem RGBA (numpy array)."""
    # Inicializa canvas vazio
    canvas = np.zeros((CANVAS_H, CANVAS_W), dtype=np.float32)

    # Posições dos sensores
    coords = SENSOR_POS_RIGHT if foot_side == "right" else {k: (1 - x, y) for k, (x, y) in SENSOR_POS_RIGHT.items()}

    # Coloca valores no canvas
    for i in range(1, 10):
        v = sr_vals.get(f"SR{i}")
        if np.isnan(v):
            continue
        x = int(coords[f"SR{i}"][0] * CANVAS_W)
        y = int(coords[f"SR{i}"][1] * CANVAS_H)
        canvas[y, x] = v

    # Aplica filtro Gaussiano para suavizar
    heatmap = gaussian_filter(canvas, sigma=SIGMA)
    # Normaliza [0,1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    heatmap = heatmap ** GAMMA

    # Converte intensidade em cor (escala HSV → RGB)
    frac = heatmap ** 0.8
    hue = (1 - frac) * 240  # 0=vermelho, 240=azul
    h = hue / 360.0
    s = np.ones_like(hue)
    v = np.ones_like(hue)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])

    # Converte para uint8 e RGBA
    r = (r * 255).astype(np.uint8)
    g = (g * 255).astype(np.uint8)
    b = (b * 255).astype(np.uint8)
    alpha = (180 + 75 * frac).astype(np.uint8)  # transparência

    return np.dstack([r, g, b, alpha])

# =========================
# INTERFACE GRÁFICA (Qt)
# =========================
class HeatmapApp(QWidget):
    def __init__(self):
        super().__init__()
        init_firebase()
        self.setWindowTitle("Visualizador de Coletas")

        # Define fonte maior
        font = self.font()
        font.setPointSize(12)
        self.setFont(font)

        # Define tema escuro
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(30, 30, 30))
        palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
        palette.setColor(QPalette.Base, QColor(40, 40, 40))
        palette.setColor(QPalette.Text, QColor(220, 220, 220))
        palette.setColor(QPalette.Button, QColor(50, 50, 50))
        palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        self.setPalette(palette)

        # Layout principal (esquerda = controles, direita = imagens)
        main_layout = QHBoxLayout(self)

        # -------- CONTROLES (lado esquerdo) --------
        left = QVBoxLayout()

        # Seleção de usuário e data
        self.combo_user = QComboBox()
        self.combo_date = QComboBox()
        btn_open = QPushButton("Abrir Heatmap")
        left.addWidget(QLabel("Usuário:"))
        left.addWidget(self.combo_user)
        left.addWidget(QLabel("Data:"))
        left.addWidget(self.combo_date)
        left.addWidget(btn_open)

        # Seletor de visualização (aparece só se existirem os dois pés)
        self.combo_view = QComboBox()
        self.combo_view.addItems(["Direito", "Esquerdo", "Ambos"])
        self.combo_view.hide()
        left.addWidget(QLabel("Visualização:"))
        left.addWidget(self.combo_view)

        # Navegação entre coletas
        nav = QHBoxLayout()
        self.btn_prev = QPushButton("⬅ Anterior")
        self.btn_next = QPushButton("Próxima ➡")
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.btn_next)
        left.addLayout(nav)

        # Status (mostra coleta atual, total e hora)
        self.status = QLabel("Status: pronto")
        left.addWidget(self.status)

        main_layout.addLayout(left, 1)

        # -------- IMAGEM (lado direito) --------
        self.img_layout = QHBoxLayout()
        self.label_img_left = QLabel("")
        self.label_img_left.setAlignment(Qt.AlignCenter)
        self.label_img_right = QLabel("")
        self.label_img_right.setAlignment(Qt.AlignCenter)
        self.img_layout.addWidget(self.label_img_left)
        self.img_layout.addWidget(self.label_img_right)
        main_layout.addLayout(self.img_layout, 3)

        # Carrega lista de usuários do banco
        self.users = list_users()
        for uid, nome in self.users:
            self.combo_user.addItem(nome, uid)

        # Conecta eventos
        self.combo_user.currentIndexChanged.connect(self.load_dates)
        btn_open.clicked.connect(self.open_heatmap)
        self.btn_prev.clicked.connect(self.prev_coleta)
        self.btn_next.clicked.connect(self.next_coleta)

        # Variáveis de estado
        self.idx = 0
        self.uid = None
        self.date = None
        self.coletasR = []
        self.coletasL = []
        self.show_left = False
        self.show_right = False

        self.resize(1200, 700)

    # ---------------- MÉTODOS ----------------
    def load_dates(self):
        """Carrega todas as datas disponíveis para o usuário selecionado (pé direito)."""
        self.combo_date.clear()
        uid = self.combo_user.currentData()
        if not uid:
            return
        for d in list_dates(uid, "R"):
            self.combo_date.addItem(d, d)

    def open_heatmap(self):
        """Abre as coletas válidas da data escolhida."""
        self.uid = self.combo_user.currentData()
        self.date = self.combo_date.currentData()
        if not self.uid or not self.date:
            QMessageBox.warning(self, "Erro", "Selecione usuário e data")
            return

        # Checa se o usuário possui InsolesL / InsolesR
        info = _get(f"Users/{self.uid}")
        self.show_left = str(info.get("InsolesL", "false")).lower() == "true"
        self.show_right = str(info.get("InsolesR", "false")).lower() == "true"

        # Carrega coletas válidas
        self.coletasR = list_coletas_validas(self.uid, self.date, "R") if self.show_right else []
        self.coletasL = list_coletas_validas(self.uid, self.date, "L") if self.show_left else []

        if not self.coletasR and not self.coletasL:
            QMessageBox.warning(self, "Erro", "Nenhuma coleta válida encontrada")
            return

        # Mostra seletor apenas se houver os dois pés
        if self.show_left and self.show_right:
            self.combo_view.show()
        else:
            self.combo_view.hide()

        self.idx = 0
        self.show_current()

    def show_current(self):
        """Mostra a coleta atual no heatmap."""
        self.label_img_left.clear()
        self.label_img_right.clear()
        # Determina modo de exibição
        view_mode = self.combo_view.currentText() if self.combo_view.isVisible() else (
            "Direito" if self.show_right else "Esquerdo"
        )

        # Texto inicial do status
        max_len = max(len(self.coletasR), len(self.coletasL))
        info_text = f"Coleta {self.idx+1}/{max_len}"

        # Renderiza pé direito
        if view_mode in ("Direito", "Ambos") and self.show_right and self.idx < len(self.coletasR):
            payloadR = get_coleta(self.uid, self.date, self.coletasR[self.idx], "R")
            if payloadR:
                self._draw_foot(payloadR)
                info_text += f" | Hora: {payloadR['timestamp']}"

        # Renderiza pé esquerdo
        if view_mode in ("Esquerdo", "Ambos") and self.show_left and self.idx < len(self.coletasL):
            payloadL = get_coleta(self.uid, self.date, self.coletasL[self.idx], "L")
            if payloadL:
                self._draw_foot(payloadL)
                info_text += f" | Hora: {payloadL['timestamp']}"

        self.status.setText(info_text)

    def _draw_foot(self, payload):
        """Desenha o pé (heatmap + máscara)."""
        heatmap = render_heatmap(payload["sr_vals"], payload["foot_side"])
        mask_path = FOOT_MASK_RIGHT if payload["foot_side"] == "right" else FOOT_MASK_LEFT
        mask = Image.open(mask_path).convert("RGBA").resize((CANVAS_W, CANVAS_H))
        img_heatmap = Image.fromarray(heatmap, mode="RGBA")
        combined = Image.alpha_composite(img_heatmap, mask)
        qimg = QImage(combined.tobytes(), combined.width, combined.height, QImage.Format_RGBA8888)
        pix = QPixmap.fromImage(qimg).scaled(500, 650, Qt.KeepAspectRatio)
        if payload["foot_side"] == "right":
            self.label_img_right.setPixmap(pix)
        else:
            self.label_img_left.setPixmap(pix)

    def keyPressEvent(self, event):
        """Permite navegar entre coletas usando ← e → do teclado."""
        if event.key() == Qt.Key_Right:
            self.next_coleta()
        elif event.key() == Qt.Key_Left:
            self.prev_coleta()

    def next_coleta(self):
        """Vai para a próxima coleta."""
        max_len = max(len(self.coletasR), len(self.coletasL))
        if max_len == 0:
            return
        self.idx = (self.idx + 1) % max_len
        self.show_current()

    def prev_coleta(self):
        """Vai para a coleta anterior."""
        max_len = max(len(self.coletasR), len(self.coletasL))
        if max_len == 0:
            return
        self.idx = (self.idx - 1) % max_len
        self.show_current()

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = HeatmapApp()
    win.show()
    sys.exit(app.exec())

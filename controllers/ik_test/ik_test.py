"""PICK AND PLACE Niryo Ned: rekam pose lewat FK slider, replay lewat IK (Jacobian).

Menggunakan modul mandiri ik_solver.py (NiryoNedIK) berbasis Standard DH
dan Geometric Jacobian (Pseudo-Inverse) dengan opsi pembanding DLS.
"""

import json
import math
import os
import tkinter as tk
from dataclasses import dataclass, asdict
from tkinter import ttk, messagebox

from controller import Robot
from ik_solver import NiryoNedIK

# =============================================================
# INISIALISASI ROBOT & SOLVER IK
# =============================================================
robot = Robot()
STEP_MS = int(robot.getBasicTimeStep())

motors = [robot.getDevice(f"joint_{i}") for i in range(1, 7)]
sensors = [robot.getDevice(f"joint_{i}_sensor") for i in range(1, 7)]

for motor, sensor in zip(motors, sensors):
    motor.setVelocity(min(0.8, motor.getMaxVelocity()))
    sensor.enable(STEP_MS)

# Inisialisasi gripper (mendukung gripper::left tunggal maupun berpasangan dengan right)
gripper_left = robot.getDevice("gripper::left")
gripper_right = robot.getDevice("gripper::right")
gripper_left.setVelocity(1.0)
if gripper_right is not None:
    gripper_right.setVelocity(1.0)

GRIPPER_BUKA = gripper_left.getMaxPosition()
GRIPPER_TUTUP = gripper_left.getMinPosition()

# Inisialisasi solver mandiri kita
ik = NiryoNedIK()
HOME = [0.0] * 6
FILE_SEKUENS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sekuens_ned.json")

# Parameter replay
TOLERANSI_JOINT = math.radians(1.0)  # Dianggap tiba jika selisih maksimum < 1 derajat
TIMEOUT_GERAK = 10.0                 # Batas waktu tunggu gerakan (detik)
JEDA_GRIPPER = 0.8                   # Waktu tunggu aksi gripper (detik)

# Rentang slider FK (derajat) diambil langsung dari joint_limits di ik_solver.py
SLIDER_DEG = [(math.degrees(lo), math.degrees(hi)) for lo, hi in ik.joint_limits]


def fk_xyz(sudut):
    """Hitung koordinat (x, y, z) frame-6 menggunakan FK dari ik_solver.py."""
    pos, _ = ik.forward_kinematics(sudut)
    return pos.tolist()


def baca_sudut():
    """Baca sudut aktual keenam joint (radian) dari PositionSensor."""
    return [sensor.getValue() for sensor in sensors]


def gerakkan_joint(sudut):
    """Kirim perintah enam sudut (radian) ke motor."""
    for motor, q in zip(motors, sudut):
        motor.setPosition(float(q))


def set_gripper(tertutup):
    """Buka atau tutup gripper."""
    pos = GRIPPER_TUTUP if tertutup else GRIPPER_BUKA
    gripper_left.setPosition(pos)
    if gripper_right is not None:
        gripper_right.setPosition(pos)


# =============================================================
# DATA: WAYPOINT & SEKUENS
# =============================================================
@dataclass
class Waypoint:
    """Satu titik pada sekuens pick and place."""
    nama: str
    xyz: list               # [x, y, z] meter, hasil FK saat direkam
    gripper_tertutup: bool  # Status gripper setelah tiba di titik ini

    def ringkas(self):
        x, y, z = self.xyz
        aksi = "TUTUP" if self.gripper_tertutup else "BUKA"
        return f"{self.nama:<6s} X={x:+.3f} Y={y:+.3f} Z={z:+.3f}  gripper:{aksi}"


class Sekuens:
    """Daftar waypoint berurutan dengan fitur simpan/muat JSON."""

    def __init__(self):
        self.titik = []

    def tambah(self, xyz, gripper_tertutup):
        nama = f"P{len(self.titik) + 1}"
        self.titik.append(Waypoint(nama, [float(v) for v in xyz], bool(gripper_tertutup)))

    def hapus(self, indeks):
        del self.titik[indeks]
        for i, wp in enumerate(self.titik, 1):
            wp.nama = f"P{i}"

    def kosongkan(self):
        self.titik.clear()

    def simpan(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump([asdict(wp) for wp in self.titik], f, indent=2)

    def muat(self, path):
        with open(path, encoding="utf-8") as f:
            self.titik = [Waypoint(**d) for d in json.load(f)]


# =============================================================
# REPLAY: STATE MACHINE BERBASIS JACOBIAN (DENGAN OPSI DLS)
# =============================================================
class Pemutar:
    """Menjalankan sekuens waypoint via IK iteratif dari ik_solver.py."""

    def __init__(self, sekuens, lapor_status, lapor_metrik):
        self.sekuens = sekuens
        self.lapor_status = lapor_status
        self.lapor_metrik = lapor_metrik
        self.aktif = False
        self.indeks = 0
        self.fase = "gerak"
        self.mulai = 0.0
        self.target_q = HOME
        self.ulang = False
        self.metode = "jacobian"

    def mulai_replay(self, metode="jacobian", ulang=False):
        if not self.sekuens.titik:
            self.lapor_status("Sekuens kosong: simpan minimal satu titik terlebih dahulu.")
            return
        self.metode = metode
        self.ulang = ulang
        self.aktif = True
        self.indeks = 0
        print(f"\n=== MULAI REPLAY ({len(self.sekuens.titik)} titik | Metode: {metode.upper()}) ===")
        self._mulai_titik()

    def berhenti(self, pesan="Replay dihentikan."):
        self.aktif = False
        self.lapor_status(pesan)

    def _mulai_titik(self):
        """Hitung IK titik saat ini menggunakan NiryoNedIK.solve_ik()."""
        wp = self.sekuens.titik[self.indeks]
        q_curr = baca_sudut()
        q_sol, ok, err, iters, min_w = ik.solve_ik(
            wp.xyz, q_curr, method=self.metode, keep_vertical=True
        )
        if not ok:
            q_sol, ok, err, iters, min_w = ik.solve_ik(
                wp.xyz, q_curr, method=self.metode, keep_vertical=False
            )

        info_metrik = (
            f"Metode: {self.metode.upper()} | Iterasi: {iters} | "
            f"Galat: {err * 1000:.2f} mm | Min Manipulabilitas (w): {min_w:.4f}"
        )
        self.lapor_metrik(info_metrik)
        print(f"[{wp.nama}] Konvergen: {ok} | {info_metrik}")

        if not ok:
            self.berhenti(f"IK GAGAL di {wp.nama} (galat {err * 1000:.1f} mm). Replay berhenti.")
            return

        self.target_q = q_sol.tolist()
        gerakkan_joint(self.target_q)
        self.fase = "gerak"
        self.mulai = robot.getTime()
        self.lapor_status(
            f"Replay {wp.nama} ({self.indeks + 1}/{len(self.sekuens.titik)}) via {self.metode.upper()} ..."
        )

    def update(self):
        if not self.aktif:
            return
        berlalu = robot.getTime() - self.mulai
        wp = self.sekuens.titik[self.indeks]

        if self.fase == "gerak":
            selisih = max(abs(a - b) for a, b in zip(baca_sudut(), self.target_q))
            if selisih < TOLERANSI_JOINT or berlalu > TIMEOUT_GERAK:
                set_gripper(wp.gripper_tertutup)
                self.fase = "gripper"
                self.mulai = robot.getTime()
        elif self.fase == "gripper" and berlalu > JEDA_GRIPPER:
            self.indeks += 1
            if self.indeks >= len(self.sekuens.titik):
                if self.ulang:
                    self.indeks = 0
                else:
                    self.aktif = False
                    self.lapor_status("Replay selesai.")
                    print("=== REPLAY SELESAI ===")
                    return
            self._mulai_titik()


# =============================================================
# GUI TKINTER
# =============================================================
class PanelPickNPlace:
    """Antarmuka Tkinter untuk rekam pose FK dan replay IK Jacobian."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Niryo Ned - Pick & Place (FK Rekam, IK Jacobian Replay)")
        self.root.geometry("660x810")
        self.root.protocol("WM_DELETE_WINDOW", self.root.iconify)

        self.sekuens = Sekuens()
        self.status = tk.StringVar(value="Mode FK: geser slider untuk atur pose, lalu klik SIMPAN TITIK.")
        self.metrik_ik = tk.StringVar(value="Metrik IK: -")
        self.posisi = tk.StringVar(value="-")
        self.gripper_tertutup = tk.BooleanVar(value=False)
        self.metode_ik = tk.StringVar(value="jacobian")
        self.ulang = tk.BooleanVar(value=False)

        self.pemutar = Pemutar(
            self.sekuens,
            lambda t: self.status.set(t),
            lambda m: self.metrik_ik.set(m),
        )

        self.vars_theta = []
        self.sedang_sinkron = False
        self.jumlah_step = 0
        self.replay_sebelumnya = False

        self._buat_slider_fk()
        self._buat_gripper_dan_rekam()
        self._buat_daftar_titik()
        self._buat_replay()

        # Panel pemantauan bawah
        ttk.Label(self.root, textvariable=self.posisi, font=("Consolas", 11, "bold")).pack(pady=(6, 2))
        ttk.Label(self.root, textvariable=self.metrik_ik, font=("Consolas", 9),
                  foreground="#2e6f40").pack(pady=(0, 4))
        ttk.Label(self.root, textvariable=self.status, wraplength=620,
                  foreground="#145b80").pack(pady=(0, 8))

        gerakkan_joint(HOME)
        set_gripper(False)
        self.root.after(1, self.langkah_simulasi)

    # ---------- penyusunan widget ----------
    def _buat_slider_fk(self):
        grup = ttk.LabelFrame(self.root, text="1. MODE FK - Slider Sudut Joint (derajat)", padding=8)
        grup.pack(fill="x", padx=12, pady=(10, 4))
        grup.columnconfigure(1, weight=1)
        for i, (lo, hi) in enumerate(SLIDER_DEG):
            var = tk.DoubleVar(value=0.0)
            self.vars_theta.append(var)
            ttk.Label(grup, text=f"θ{i + 1}", width=4).grid(row=i, column=0)
            slider = tk.Scale(
                grup, from_=round(lo + 0.5, 1), to=round(hi - 0.5, 1),
                resolution=0.1, orient="horizontal", variable=var,
                length=430, highlightthickness=0, command=self.slider_fk_berubah
            )
            slider.grid(row=i, column=1, sticky="ew", padx=4)

    def _buat_gripper_dan_rekam(self):
        grup = ttk.LabelFrame(self.root, text="2. Gripper & Rekam Titik", padding=8)
        grup.pack(fill="x", padx=12, pady=4)
        ttk.Radiobutton(grup, text="Gripper BUKA", variable=self.gripper_tertutup,
                        value=False, command=self.ubah_gripper).pack(side="left", padx=4)
        ttk.Radiobutton(grup, text="Gripper TUTUP", variable=self.gripper_tertutup,
                        value=True, command=self.ubah_gripper).pack(side="left", padx=4)
        ttk.Button(grup, text="SIMPAN TITIK", command=self.simpan_titik).pack(side="right", padx=4)

    def _buat_daftar_titik(self):
        grup = ttk.LabelFrame(self.root, text="3. Sekuens Titik (Waypoint)", padding=8)
        grup.pack(fill="both", expand=True, padx=12, pady=4)
        self.listbox = tk.Listbox(grup, height=7, font=("Consolas", 10), exportselection=False)
        self.listbox.pack(fill="both", expand=True)
        baris = ttk.Frame(grup)
        baris.pack(fill="x", pady=(6, 0))
        ttk.Button(baris, text="Hapus terpilih", command=self.hapus_titik).pack(side="left", padx=2)
        ttk.Button(baris, text="Kosongkan", command=self.kosongkan).pack(side="left", padx=2)
        ttk.Button(baris, text="Simpan JSON", command=self.simpan_file).pack(side="right", padx=2)
        ttk.Button(baris, text="Muat JSON", command=self.muat_file).pack(side="right", padx=2)

    def _buat_replay(self):
        grup = ttk.LabelFrame(self.root, text="4. REPLAY dengan IK (ik_solver.py)", padding=8)
        grup.pack(fill="x", padx=12, pady=4)

        baris_metode = ttk.Frame(grup)
        baris_metode.pack(fill="x", pady=(0, 6))
        ttk.Label(baris_metode, text="Metode IK:").pack(side="left", padx=(0, 6))
        ttk.Radiobutton(baris_metode, text="Jacobian (Utama)",
                        variable=self.metode_ik, value="jacobian").pack(side="left", padx=4)
        ttk.Radiobutton(baris_metode, text="DLS (Pembanding)",
                        variable=self.metode_ik, value="dls").pack(side="left", padx=4)

        baris_tombol = ttk.Frame(grup)
        baris_tombol.pack(fill="x")
        ttk.Button(baris_tombol, text="▶ REPLAY", command=self.replay).pack(side="left", padx=2)
        ttk.Button(baris_tombol, text="■ STOP", command=self.stop).pack(side="left", padx=2)
        ttk.Button(baris_tombol, text="HOME", command=self.home).pack(side="left", padx=2)
        ttk.Checkbutton(baris_tombol, text="Ulangi terus", variable=self.ulang).pack(side="left", padx=10)

    # ---------- MODE FK ----------
    def sudut_slider(self):
        return [math.radians(v.get()) for v in self.vars_theta]

    def slider_fk_berubah(self, _nilai=None):
        if self.pemutar.aktif or self.sedang_sinkron:
            return
        gerakkan_joint(self.sudut_slider())

    def ubah_gripper(self):
        if not self.pemutar.aktif:
            set_gripper(self.gripper_tertutup.get())

    def sinkron_slider(self, sudut):
        self.sedang_sinkron = True
        for var, q in zip(self.vars_theta, sudut):
            var.set(round(math.degrees(q), 1))
        self.sedang_sinkron = False

    # ---------- rekam ----------
    def simpan_titik(self):
        if self.pemutar.aktif:
            return
        xyz = fk_xyz(self.sudut_slider())
        self.sekuens.tambah(xyz, self.gripper_tertutup.get())
        self.refresh_daftar()
        self.status.set(f"Titik {self.sekuens.titik[-1].nama} disimpan.")

    def hapus_titik(self):
        pilihan = self.listbox.curselection()
        if pilihan and not self.pemutar.aktif:
            self.sekuens.hapus(pilihan[0])
            self.refresh_daftar()

    def kosongkan(self):
        if not self.pemutar.aktif:
            self.sekuens.kosongkan()
            self.refresh_daftar()

    def refresh_daftar(self):
        self.listbox.delete(0, "end")
        for wp in self.sekuens.titik:
            self.listbox.insert("end", wp.ringkas())

    def simpan_file(self):
        self.sekuens.simpan(FILE_SEKUENS)
        self.status.set(f"Sekuens disimpan ke {FILE_SEKUENS}")

    def muat_file(self):
        try:
            self.sekuens.muat(FILE_SEKUENS)
        except (OSError, ValueError, TypeError) as exc:
            messagebox.showerror("Gagal memuat", str(exc))
            return
        self.refresh_daftar()
        self.status.set(f"Memuat {len(self.sekuens.titik)} titik dari file.")

    # ---------- replay ----------
    def replay(self):
        if not self.pemutar.aktif:
            self.pemutar.mulai_replay(
                metode=self.metode_ik.get(),
                ulang=self.ulang.get()
            )

    def stop(self):
        if self.pemutar.aktif:
            self.pemutar.berhenti()

    def home(self):
        self.stop()
        self.sinkron_slider(HOME)
        gerakkan_joint(HOME)
        self.status.set("Robot kembali ke HOME.")

    # ---------- loop simulasi ----------
    def perbarui_tampilan(self):
        x, y, z = fk_xyz(baca_sudut())
        self.posisi.set(f"XYZ aktual (DH): X={x:+.3f}  Y={y:+.3f}  Z={z:+.3f}")

    def langkah_simulasi(self):
        if robot.step(STEP_MS) == -1:
            self.root.destroy()
            return
        self.pemutar.update()
        if self.replay_sebelumnya and not self.pemutar.aktif:
            self.sinkron_slider(baca_sudut())
        self.replay_sebelumnya = self.pemutar.aktif
        self.jumlah_step += 1
        if self.jumlah_step % max(1, round(100 / STEP_MS)) == 0:
            self.perbarui_tampilan()
        self.root.after(1, self.langkah_simulasi)


if __name__ == "__main__":
    PanelPickNPlace().root.mainloop()
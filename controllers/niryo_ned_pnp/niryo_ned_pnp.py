"""
niryo_ned_pnp.py
================
Controller Pick and Place Niryo Ned di simulator Webots (R2025a).

Alur kerja per kotak (Finite State Machine):
    PRE_PICK  ->  PICK  ->  GRASP  ->  LIFT  ->  TRANSIT
      ->  PRE_PLACE  ->  PLACE  ->  RELEASE  ->  RETRACT

Cara menambah kasus Pick and Place baru:
Cukup tambahkan dictionary baru ke dalam daftar `TASKS` di bawah ini
dengan koordinat `pick` dan `place` dalam frame World Webots (meter).
"""

import os
import numpy as np
from controller import Supervisor
from ik_solver import NiryoNedIK


# =============================================================================
# KONFIGURASI UTAMA (MUDAH DIUBAH / DITAMBAH UNTUK KASUS BARU)
# =============================================================================

# Metode IK: "dls" (Damped Least Squares) atau "jacobian" (Moore-Penrose)
METHOD = "jacobian"

# Posisi & rotasi base robot di World Webots (biarkan 0 jika robot di origin)
ROBOT_BASE_POS = np.array([0.0, 0.0, 0.0], dtype=float)
ROBOT_BASE_YAW = 0.0

# Kompensasi tambahan sumbu Z TCP (meter) dan tinggi angkat aman (meter)
GRIPPER_Z_OFFSET = 0.0
Z_OFFSET = 0.10

# Titik transit aman di depan robot saat menyeberang dari zona pick (+X) ke place (-X)
SAFE_WAYPOINT = np.array([0.0, 0.22, 0.15], dtype=float)

# Daftar tugas Pick & Place (urut dari atas ke bawah).
# Catatan untuk kotak 2x2x2 cm (pusat z=0.01 m) & area tebal 0.5 cm:
# - pick  z = 0.019 m agar ujung jari gripper mencengkeram kotak tanpa menyentuh lantai.
# - place z = 0.024 m agar kotak diletakkan tepat di atas area tanpa menekan lantai.
TASKS = [
    {
        "name": "Green Box -> Green Area",
        "def_name": "GREEN_BOX",
        "pick":  [0.24,  0.10, 0.019],
        "place": [-0.24, 0.14, 0.024],
    },
    {
        "name": "Red Box -> Red Area",
        "def_name": "RED_BOX",
        "pick":  [0.24,  0.00, 0.019],
        "place": [-0.24, 0.00, 0.024],
    },
    {
        "name": "Blue Box -> Blue Area",
        "def_name": "BLUE_BOX",
        "pick":  [0.24, -0.10, 0.019],
        "place": [-0.24, -0.14, 0.024],
    },
    # {
    #     "name": "White Box -> Blue Area",
    #     "def_name": "WHITE_BOX",
    #     "pick":  [0.15, 0.20, 0.019],
    #     "place": [-0.24, -0.14, 0.04],
    # },
]

# Parameter motor, gripper, dan toleransi gerak
MOTOR_VELOCITY = 0.5        # Kecepatan sendi lengan (rad/s) agar kotak tidak tersentak
GRIPPER_SPEED = 0.01        # Kecepatan rahang gripper (m/s) agar jepitan stabil
GRIPPER_OPEN_POS = 0.01     # Posisi buka penuh (m)
GRIPPER_CLOSE_POS = -0.01   # Posisi jepit penuh untuk kotak 2 cm (m)
GRASP_WAIT_STEPS = 80       # Jeda agar rahang mencengkeram kotak dengan mantap
RELEASE_WAIT_STEPS = 25     # Jeda setelah rahang dibuka sebelum lengan naik
REACH_THRESHOLD = 0.008     # Toleransi norma selisih sudut untuk dianggap tiba (rad)
MAX_WAIT_STEPS = 300        # Batas step per gerakan agar FSM tidak macet
IK_TOLERANCE = 5e-4         # Toleransi konvergensi IK (0.5 mm)


# =============================================================================
# FUNGSI BANTU TRANSFORMASI KOORDINAT
# =============================================================================

def world_to_dh_frame(pos_world):
    """Konversi koordinat World Webots ke frame lokal base robot + offset Z TCP."""
    p_rel = np.asarray(pos_world, dtype=float) - ROBOT_BASE_POS
    c, s = np.cos(-ROBOT_BASE_YAW), np.sin(-ROBOT_BASE_YAW)
    rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    p_dh = rot_z @ p_rel
    p_dh[2] += GRIPPER_Z_OFFSET
    return p_dh


# =============================================================================
# CONTROLLER UTAMA (FINITE STATE MACHINE)
# =============================================================================

def main():
    robot = Supervisor()
    timestep = int(robot.getBasicTimeStep())
    auto_quit = os.environ.get("WEBOTS_AUTO_QUIT") == "1"
    method = os.environ.get("IK_METHOD", METHOD)
    is_supervisor = robot.getSelf() is not None

    # 1. Inisialisasi 6 motor & sensor sendi lengan
    motors = []
    sensors = []
    for i in range(1, 7):
        motor = robot.getDevice(f"joint_{i}")
        motor.setVelocity(min(MOTOR_VELOCITY, motor.getMaxVelocity()))
        motors.append(motor)

        sensor = robot.getDevice(f"joint_{i}_sensor")
        sensor.enable(timestep)
        sensors.append(sensor)

    # 2. Inisialisasi gripper kiri & kanan
    gripper_left = robot.getDevice("gripper::left")
    gripper_right = robot.getDevice("gripper::right")
    for g in (gripper_left, gripper_right):
        if g is not None:
            g.setVelocity(GRIPPER_SPEED)

    def set_gripper(open_gripper=True):
        pos = GRIPPER_OPEN_POS if open_gripper else GRIPPER_CLOSE_POS
        if gripper_left is not None:
            gripper_left.setPosition(pos)
        if gripper_right is not None:
            gripper_right.setPosition(pos)

    def get_current_joints():
        return np.array([s.getValue() for s in sensors], dtype=float)

    def set_target_joints(q_target):
        for motor, val in zip(motors, q_target):
            motor.setPosition(float(val))

    def get_box_world_pos(def_name):
        """Ambil posisi fisik kotak via Supervisor (jika aktif) untuk verifikasi."""
        if not is_supervisor or not def_name:
            return None
        node = robot.getFromDef(def_name)
        return np.array(node.getPosition(), dtype=float) if node is not None else None

    # 3. Siapkan daftar target dalam frame robot
    tasks_dh = [
        {
            "name": task["name"],
            "def_name": task.get("def_name"),
            "pick": world_to_dh_frame(task["pick"]),
            "place": world_to_dh_frame(task["place"]),
            "place_world": np.array(task["place"], dtype=float),
        }
        for task in TASKS
    ]
    transit_dh = world_to_dh_frame(SAFE_WAYPOINT)
    lift_vec = np.array([0.0, 0.0, Z_OFFSET], dtype=float)

    # 4. Inisialisasi solver IK & state FSM
    ik = NiryoNedIK()
    metrics_log = []
    task_idx = 0
    state = "INIT"
    q_target = np.zeros(6, dtype=float)
    wait_steps = 0
    move_timeout = 0
    initial_box_z = None

    def command_move(target_dh, label):
        """Hitung IK dengan orientasi gripper vertikal lalu gerakkan motor."""
        nonlocal q_target, move_timeout, state
        q_curr = get_current_joints()
        q_sol, ok, err, iters, min_w = ik.solve_ik(
            target_dh, q_curr, method=method, tol=IK_TOLERANCE, keep_vertical=True
        )
        metrics_log.append((ok, err * 1000.0, iters, min_w))

        task_name = tasks_dh[task_idx]["name"]
        print(
            f"[{task_name} | {label:9s}] "
            f"Target: {np.round(target_dh, 3)} | OK: {str(ok):5s} | "
            f"Iter: {iters:2d} | Err: {err * 1000:5.2f} mm | Manip: {min_w:.4f}",
            flush=True,
        )

        if not ok:
            print(f"  [ABORT] Target {label} gagal dicapai (err={err * 1000:.1f} mm).", flush=True)
            state = "ERROR"
            return False

        q_target = q_sol
        set_target_joints(q_target)
        move_timeout = MAX_WAIT_STEPS
        return True

    def arrived():
        """Cek apakah sendi sudah mencapai target atau melewati batas tunggu."""
        nonlocal move_timeout
        move_timeout -= 1
        err_q = float(np.linalg.norm(get_current_joints() - q_target))
        if err_q < REACH_THRESHOLD:
            return True
        if move_timeout <= 0:
            print(f"  [TIMEOUT] Sisa selisih sendi: {err_q:.4f} rad, lanjut.", flush=True)
            return True
        return False

    def print_summary():
        if not metrics_log:
            return
        total = len(metrics_log)
        sukses = sum(1 for item in metrics_log if item[0])
        avg_err = np.mean([item[1] for item in metrics_log])
        avg_iter = np.mean([item[2] for item in metrics_log])
        min_w = np.min([item[3] for item in metrics_log])
        print("\n================ RINGKASAN PERFORMA IK ================", flush=True)
        print(f"Metode IK               : {method.upper()}", flush=True)
        print(f"Waypoint Konvergen      : {sukses} / {total}", flush=True)
        print(f"Rata-rata Galat Posisi  : {avg_err:.2f} mm", flush=True)
        print(f"Rata-rata Iterasi       : {avg_iter:.1f}", flush=True)
        print(f"Manipulabilitas Minimum : {min_w:.5f}", flush=True)
        print("=======================================================\n", flush=True)

    # Langkah awal agar sensor terisi dan gripper terbuka siap
    robot.step(timestep)
    set_gripper(open_gripper=True)
    print(f"=== Memulai Pick and Place {len(tasks_dh)} Kotak (Metode: {method.upper()}) ===", flush=True)

    # 5. Main Loop FSM
    while robot.step(timestep) != -1:

        if state == "INIT":
            if task_idx >= len(tasks_dh):
                set_target_joints(np.zeros(6, dtype=float))
                q_target = np.zeros(6, dtype=float)
                move_timeout = MAX_WAIT_STEPS
                state = "GO_HOME"
            else:
                task = tasks_dh[task_idx]
                pos_now = get_box_world_pos(task["def_name"])
                initial_box_z = pos_now[2] if pos_now is not None else None
                if command_move(task["pick"] + lift_vec, "PRE_PICK"):
                    state = "PRE_PICK"

        elif state == "PRE_PICK":
            if arrived():
                if command_move(tasks_dh[task_idx]["pick"], "PICK"):
                    state = "PICK"

        elif state == "PICK":
            if arrived():
                set_gripper(open_gripper=False)
                wait_steps = GRASP_WAIT_STEPS
                state = "GRASP"

        elif state == "GRASP":
            wait_steps -= 1
            if wait_steps <= 0:
                if command_move(tasks_dh[task_idx]["pick"] + lift_vec, "LIFT"):
                    state = "LIFT"

        elif state == "LIFT":
            if arrived():
                task = tasks_dh[task_idx]
                pos_lifted = get_box_world_pos(task["def_name"])
                if pos_lifted is not None and initial_box_z is not None:
                    lifted_mm = (pos_lifted[2] - initial_box_z) * 1000.0
                    print(f"  [CEK ANGKAT] {task['def_name']} naik {lifted_mm:.1f} mm (z={pos_lifted[2]:.3f} m)", flush=True)
                    if lifted_mm < 40.0:
                        print(f"  [GAGAL] {task['def_name']} gagal terangkat!", flush=True)
                        state = "ERROR"
                        continue
                if command_move(transit_dh, "TRANSIT"):
                    state = "TRANSIT"

        elif state == "TRANSIT":
            if arrived():
                if command_move(tasks_dh[task_idx]["place"] + lift_vec, "PRE_PLACE"):
                    state = "PRE_PLACE"

        elif state == "PRE_PLACE":
            if arrived():
                if command_move(tasks_dh[task_idx]["place"], "PLACE"):
                    state = "PLACE"

        elif state == "PLACE":
            if arrived():
                set_gripper(open_gripper=True)
                wait_steps = RELEASE_WAIT_STEPS
                state = "RELEASE"

        elif state == "RELEASE":
            wait_steps -= 1
            if wait_steps <= 0:
                if command_move(tasks_dh[task_idx]["place"] + lift_vec, "RETRACT"):
                    state = "RETRACT"

        elif state == "RETRACT":
            if arrived():
                task = tasks_dh[task_idx]
                pos_end = get_box_world_pos(task["def_name"])
                if pos_end is not None:
                    err_xy_mm = np.linalg.norm(pos_end[:2] - task["place_world"][:2]) * 1000.0
                    print(
                        f"  [CEK TARUH] {task['def_name']} di {np.round(pos_end, 4)} | "
                        f"Galat XY ke target area: {err_xy_mm:.1f} mm",
                        flush=True,
                    )
                    if err_xy_mm > 30.0:
                        print(f"  [GAGAL] {task['def_name']} melenceng > 30 mm!", flush=True)
                        state = "ERROR"
                        continue
                task_idx += 1
                state = "INIT"

        elif state == "GO_HOME":
            if arrived():
                print("=== Seluruh Kotak Berhasil Dipindahkan ke Area Warnanya! ===", flush=True)
                print_summary()
                state = "DONE"
                wait_steps = 5

        elif state == "DONE":
            if auto_quit:
                wait_steps -= 1
                if wait_steps <= 0:
                    robot.simulationQuit(0)
                    break

        elif state == "ERROR":
            set_target_joints(get_current_joints())
            print_summary()
            state = "HALT"
            wait_steps = 5

        elif state == "HALT":
            if auto_quit:
                wait_steps -= 1
                if wait_steps <= 0:
                    robot.simulationQuit(1)
                    break


if __name__ == "__main__":
    main()

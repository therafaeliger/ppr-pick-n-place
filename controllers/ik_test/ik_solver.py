"""
ik_solver.py
============
Modul Forward Kinematics (FK), Geometric Jacobian, dan Inverse Kinematics (IK)
untuk robot manipulator 6-DOF Niryo Ned (Webots R2025a, Ned.proto).

Konsep utama yang diimplementasikan (untuk keperluan pembelajaran & presentasi):
1. Standard Denavit-Hartenberg (DH) yang terkalibrasi terhadap geometri fisik
   Ned.proto Webots R2025a hingga titik tengah celah gripper (TCP).
2. Geometric Position Jacobian (3x6) untuk memetakan kecepatan sendi ke kecepatan
   linear TCP.
3. Indeks Manipulabilitas Yoshikawa w = sqrt(det(J * J^T)) untuk mendeteksi
   kedekatan konfigurasi robot terhadap singularitas.
4. Dua metode penyelesaian numerik IK untuk studi komparasi:
   - "dls"      : Damped Least Squares (Levenberg-Marquardt) dengan damping adaptif.
   - "jacobian" : Moore-Penrose Pseudo-Inverse Jacobian (J^+).
5. Mode `keep_vertical=True` untuk Pick and Place agar gripper selalu tegak lurus
   menghadap ke bawah (q4 = 0, q6 = 0, q5 = q2 + q3 - pi/2) sehingga aman saat
   mengambil dan meletakkan kotak.
"""

import numpy as np


class NiryoNedIK:
    """Solver Forward & Inverse Kinematics posisi 3D TCP untuk robot Niryo Ned."""

    def __init__(self):
        # ---------------------------------------------------------------------
        # 1. Parameter Standard DH terkalibrasi ke TCP (Ned.proto Webots R2025a)
        #    Format tiap baris: [a (m), alpha (rad), d (m), theta_offset (rad), theta_sign]
        #    Sudut DH dihitung sebagai: theta_i = theta_sign * q[i] + theta_offset
        # ---------------------------------------------------------------------
        beta2 = float(np.arctan2(0.012, 0.221))
        len_l2 = float(np.hypot(0.221, 0.012))
        beta5 = float(np.arctan2(0.0168, 0.1042))
        len_l5 = float(np.hypot(0.1042, 0.0168))

        self.dh_params = [
            [0.0,     -np.pi / 2, 0.1715,  np.pi / 2,         1.0],   # Joint 1 (Base)
            [len_l2,   0.0,       0.0,    -np.pi / 2 + beta2, 1.0],   # Joint 2 (Shoulder)
            [0.032,   -np.pi / 2, 0.0,    -beta2,             1.0],   # Joint 3 (Elbow)
            [0.0,      np.pi / 2, 0.230,   0.0,               1.0],   # Joint 4 (Forearm)
            [0.0,     -np.pi / 2, 0.0,    -beta5,            -1.0],   # Joint 5 (Wrist pitch)
            [-0.0007,  0.0,       len_l5,  0.0,              -1.0],   # Joint 6 (Hand -> TCP)
        ]

        # ---------------------------------------------------------------------
        # 2. Batas sudut sendi (Joint Limits) dari Ned.proto dalam radian
        # ---------------------------------------------------------------------
        self.joint_limits = np.array([
            [-2.8,  2.8],   # joint_1
            [-0.8,  0.8],   # joint_2
            [-1.5,  1.3],   # joint_3
            [-2.0,  2.0],   # joint_4
            [-1.5,  1.5],   # joint_5
            [-2.5,  2.5],   # joint_6
        ], dtype=float)

        # Margin kecil agar sudut hasil IK tidak mentok keras pada limit motor Webots
        self.limit_margin = 1e-5

    # =========================================================================
    # BAGIAN 1: KINEMATIKA MAJU (FORWARD KINEMATICS)
    # =========================================================================

    def dh_matrix(self, a, alpha, d, theta):
        """
        Membentuk matriks transformasi homogen 4x4 konvensi Standard DH:
            T_i = Rot_z(theta) * Trans_z(d) * Trans_x(a) * Rot_x(alpha)
        """
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            [ct,  -st * ca,   st * sa,  a * ct],
            [st,   ct * ca,  -ct * sa,  a * st],
            [0.0,  sa,        ca,       d],
            [0.0,  0.0,       0.0,      1.0],
        ], dtype=float)

    def forward_kinematics(self, q):
        """
        Menghitung posisi (x, y, z) TCP terhadap base robot (frame-0)
        serta menyimpan daftar transformasi kumulatif T_0^0 .. T_0^6.
        """
        if len(q) != 6:
            raise ValueError("Diperlukan tepat 6 sudut sendi untuk Niryo Ned.")

        T = np.eye(4, dtype=float)
        transforms = [T.copy()]

        for i in range(6):
            a, alpha, d, theta_offset, theta_sign = self.dh_params[i]
            theta = theta_sign * float(q[i]) + theta_offset
            T = T @ self.dh_matrix(a, alpha, d, theta)
            transforms.append(T.copy())

        return T[:3, 3].copy(), transforms

    # =========================================================================
    # BAGIAN 2: GEOMETRIC JACOBIAN & MANIPULABILITAS
    # =========================================================================

    def compute_position_jacobian(self, q):
        """
        Menghitung Geometric Position Jacobian (3x6) untuk 6 sendi revolute:
            J_v,i = (theta_sign_i * z_{i-1}) x (p_end - p_{i-1})
        """
        p_end, transforms = self.forward_kinematics(q)
        J = np.zeros((3, 6), dtype=float)

        for i in range(6):
            theta_sign = self.dh_params[i][4]
            z_prev = theta_sign * transforms[i][:3, 2]
            p_prev = transforms[i][:3, 3]
            J[:, i] = np.cross(z_prev, p_end - p_prev)

        return p_end, J

    @staticmethod
    def compute_manipulability(J):
        """
        Menghitung Indeks Manipulabilitas Yoshikawa:
            w = sqrt(det(J * J^T))
        """
        JJT = J @ J.T
        return float(np.sqrt(max(0.0, np.linalg.det(JJT))))

    # =========================================================================
    # BAGIAN 3: UTILITAS BATAS SENDI, ORIENTASI VERTIKAL & WORKSPACE
    # =========================================================================

    def clamp_joints(self, q):
        """Memotong sudut sendi agar tetap berada di dalam limit motor + margin aman."""
        low = self.joint_limits[:, 0] + self.limit_margin
        high = self.joint_limits[:, 1] - self.limit_margin
        return np.clip(np.asarray(q, dtype=float), low, high)

    def apply_vertical_constraint(self, q):
        """
        Menjaga gripper tegak lurus menghadap ke bawah:
            q4 = 0, q6 = 0, dan q5 = (q2 + q3) - pi/2
        """
        q_out = self.clamp_joints(q)
        q_out[3] = 0.0
        q_out[5] = 0.0
        lo5 = self.joint_limits[4, 0] + self.limit_margin
        hi5 = self.joint_limits[4, 1] - self.limit_margin
        q_out[4] = float(np.clip((q_out[1] + q_out[2]) - np.pi / 2, lo5, hi5))
        return q_out

    @staticmethod
    def is_within_rough_workspace(target_pos):
        """Pengecekan cepat batas kasar workspace Niryo Ned."""
        r_xy = float(np.hypot(target_pos[0], target_pos[1]))
        return (r_xy <= 0.65) and (-0.15 <= target_pos[2] <= 0.70)

    # =========================================================================
    # BAGIAN 4: ITERASI NUMERIK IK (SINGLE SEED + BACKTRACKING LINE SEARCH)
    # =========================================================================

    def _effective_jacobian(self, J, keep_vertical):
        """Jacobian efektif untuk mode vertikal (q5 = q2 + q3 - pi/2) atau 6-DOF bebas."""
        if not keep_vertical:
            return J
        J_eff = np.zeros((3, 6), dtype=float)
        J_eff[:, 0] = J[:, 0]
        J_eff[:, 1] = J[:, 1] + J[:, 4]
        J_eff[:, 2] = J[:, 2] + J[:, 4]
        return J_eff

    def _run_iteration(self, target_pos, q_seed, method, damping, tol, max_iter, keep_vertical):
        """Menjalankan iterasi IK dari satu tebakan awal (seed) dengan line search."""
        q = self.apply_vertical_constraint(q_seed) if keep_vertical else self.clamp_joints(q_seed)
        pos_init, J_init = self.compute_position_jacobian(q)
        err_init = float(np.linalg.norm(target_pos - pos_init))
        min_manip = self.compute_manipulability(self._effective_jacobian(J_init, keep_vertical))

        best_err = err_init
        best_q = q.copy()
        best_iter = 0
        best_min_manip = min_manip

        for iteration in range(1, max_iter + 1):
            pos_curr, J = self.compute_position_jacobian(q)
            J_use = self._effective_jacobian(J, keep_vertical)
            error = target_pos - pos_curr
            err_norm = float(np.linalg.norm(error))

            manip = self.compute_manipulability(J_use)
            min_manip = min(min_manip, manip)

            if err_norm < best_err:
                best_err = err_norm
                best_q = q.copy()
                best_iter = iteration
                best_min_manip = min_manip

            if err_norm <= tol:
                return q, True, err_norm, iteration, min_manip

            if method == "dls":
                lam = (0.045 if err_norm > 0.015 else 0.020) if damping is None else float(damping)
                A = (J_use @ J_use.T) + (lam ** 2) * np.eye(3)
                try:
                    v = np.linalg.solve(A, error)
                except np.linalg.LinAlgError:
                    break
                dq = J_use.T @ v

            elif method == "jacobian":
                dq = np.linalg.pinv(J_use, rcond=1e-4) @ error

            else:
                raise ValueError("Method harus 'dls' atau 'jacobian'.")

            if keep_vertical:
                dq[3] = 0.0
                dq[4] = dq[1] + dq[2]
                dq[5] = 0.0

            max_dq = float(np.max(np.abs(dq)))
            if max_dq > 0.2:
                dq = dq * (0.2 / max_dq)

            accepted = False
            for gain in (1.0, 0.5, 0.25, 0.125, 0.0625):
                q_cand = q + gain * dq
                q_trial = self.apply_vertical_constraint(q_cand) if keep_vertical else self.clamp_joints(q_cand)
                pos_trial, _ = self.forward_kinematics(q_trial)
                err_trial = float(np.linalg.norm(target_pos - pos_trial))

                if err_trial < err_norm - 1e-9:
                    q = q_trial
                    accepted = True
                    break

            if not accepted:
                break

        return best_q, (best_err <= tol), best_err, best_iter, best_min_manip

    # =========================================================================
    # BAGIAN 5: SOLVER UTAMA MULTI-SEED
    # =========================================================================

    @staticmethod
    def _wrap_angle(angle):
        """Bungkus sudut ke rentang [-pi, pi]."""
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _generate_seeds(self, target_pos, q_current, keep_vertical):
        """
        Menghasilkan daftar tebakan awal (multi-seed).
        Pada q1 = 0, lengan Ned mengarah ke +Y (azimut +pi/2), sehingga sudut base
        untuk mengarah langsung ke target adalah azimut - pi/2.
        """
        az = float(np.arctan2(target_pos[1], target_pos[0]))
        q1_front = self._wrap_angle(az - np.pi / 2)
        q1_back = self._wrap_angle(az + np.pi / 2)
        q_curr_list = list(q_current)

        seeds = [q_curr_list]

        if keep_vertical:
            for q2, q3 in ((0.5, 0.4), (0.2, 0.6), (0.7, 0.3), (-0.2, 0.7), (0.6, -0.2)):
                seeds.append([q1_front, q2, q3, 0.0, (q2 + q3) - np.pi / 2, 0.0])
            seeds.append([0.0] * 6)
        else:
            seeds.append([0.0] * 6)
            for q2, q3 in ((0.45, 0.45), (0.2, 0.6), (-0.45, 0.85), (0.45, -0.85)):
                for q1 in (q1_front, q1_back):
                    seeds.append([q1, q2, q3, 0.0, 0.0, 0.0])
            for q5 in (-0.8, 0.8):
                seeds.append([q_curr_list[0], q_curr_list[1], q_curr_list[2], 0.0, q5, 0.0])

        return seeds

    def solve_ik(
        self,
        target_pos,
        q_current,
        method="dls",
        damping=None,
        tol=2e-3,
        max_iter=150,
        keep_vertical=True,
    ):
        """
        Menyelesaikan Inverse Kinematics posisi 3D TCP (x, y, z) dalam frame base robot.

        Return:
            tuple (q_sol, converged, err_norm, iterations, min_manip)
        """
        target_pos = np.asarray(target_pos, dtype=float)
        if target_pos.shape != (3,) or not np.all(np.isfinite(target_pos)):
            raise ValueError("target_pos harus berupa 3 angka [x, y, z] yang valid.")

        q_curr = np.asarray(q_current, dtype=float)
        if q_curr.shape != (6,) or not np.all(np.isfinite(q_curr)):
            raise ValueError("q_current harus berupa 6 sudut sendi yang valid.")

        q_curr = self.clamp_joints(q_curr)

        if not self.is_within_rough_workspace(target_pos):
            pos_curr, J_curr = self.compute_position_jacobian(q_curr)
            err_curr = float(np.linalg.norm(target_pos - pos_curr))
            return q_curr, False, err_curr, 0, self.compute_manipulability(J_curr)

        seeds = self._generate_seeds(target_pos, q_curr, keep_vertical)

        best_overall = None
        successes = []

        for seed in seeds:
            q_sol, ok, err, iters, min_w = self._run_iteration(
                target_pos, seed, method, damping, tol, max_iter, keep_vertical
            )

            delta_q_sq = float(np.sum((q_sol - q_curr) ** 2))
            key = (err, delta_q_sq)

            if best_overall is None or key < best_overall[0]:
                best_overall = (key, (q_sol, ok, err, iters, min_w))

            if ok:
                successes.append((delta_q_sq, (q_sol, ok, err, iters, min_w)))
                if len(successes) >= 3:
                    break

        if successes:
            _, chosen = min(successes, key=lambda item: item[0])
            return chosen

        return best_overall[1]

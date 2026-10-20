"""Validated CSV joint playback with a horizontal right arm and closed hand."""
import csv
from pathlib import Path

import mujoco
import numpy as np


class RightArmTrajectory:
    names = ("RShoulderPitch", "RShoulderRoll", "RElbowYaw", "RElbowRoll", "RWristYaw", "RHand")

    def __init__(self, path, model, data, speed=1.0, loop=False):
        self.path = Path(path).resolve()
        with self.path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        values = np.array([[float(r[k]) for k in ("time_s", "RShoulderRoll_deg", "RElbowRoll_deg")] for r in rows])
        if len(values) < 2 or not np.isfinite(values).all():
            raise ValueError("CSV requires at least two finite trajectory samples")
        if values[0, 0] != 0 or np.any(np.diff(values[:, 0]) <= 0):
            raise ValueError("CSV time must start at zero and increase strictly")
        self.times, self.angles = values[:, 0], np.deg2rad(values[:, 1:])
        self.speed, self.loop = speed, loop
        self.duration = self.times[-1] / speed
        self.ids = np.array([model.actuator(n).id for n in self.names])
        self.qadr = np.array([model.joint(n).qposadr[0] for n in self.names])
        self.dadr = np.array([model.joint(n).dofadr[0] for n in self.names])
        self.targets = np.zeros((len(values), 6))
        self.targets[:, 1] = self.angles[:, 0]
        self.targets[:, 3] = self.angles[:, 1]
        self.targets[:, 4] = np.pi / 2
        for k, name in enumerate(self.names):
            limits = model.joint(name).range
            if np.any(self.targets[:, k] < limits[0]) or np.any(self.targets[:, k] > limits[1]):
                raise ValueError(f"Trajectory exceeds {name} limits")
        # Stronger arm servos while retaining the URDF actuator force limits.
        model.actuator_gainprm[self.ids, 0] = 80
        model.actuator_biasprm[self.ids, 1] = -80
        model.actuator_biasprm[self.ids, 2] = -3
        data.qpos[self.qadr] = self.targets[0]
        data.qvel[:] = 0
        data.ctrl[self.ids] = self.targets[0]
        # Initialize imported finger mimic constraints consistently with RHand=0.
        for i in range(model.neq):
            if model.eq_type[i] == mujoco.mjtEq.mjEQ_JOINT:
                j1, j2 = model.eq_obj1id[i], model.eq_obj2id[i]
                if j2 >= 0:
                    q2 = data.qpos[model.jnt_qposadr[j2]]
                    data.qpos[model.jnt_qposadr[j1]] = np.polynomial.polynomial.polyval(q2, model.eq_data[i, :5])
        mujoco.mj_forward(model, data)
        self.reference = self.targets[0].copy()
        self.max_errors = np.zeros(6)
        self.max_height_error = 0.0
        self.max_wrist_vertical_error = 0.0
        self.completed = False

    def sample(self, time_s):
        t = time_s * self.speed
        end = self.times[-1]
        # Hold endpoint 1 s, smoothly return over 2 s, hold start 1 s.
        if self.loop:
            t %= end + 4.0
            if t > end:
                u = np.clip((t - end - 1.0) / 2.0, 0, 1)
                blend = 10*u**3 - 15*u**4 + 6*u**5
                rate = (30*u**2 - 60*u**3 + 30*u**4) / 2 * self.speed
                delta = self.targets[0] - self.targets[-1]
                return self.targets[-1] + blend * delta, rate * delta
        if t >= end:
            return self.targets[-1].copy(), np.zeros(6)
        index = min(np.searchsorted(self.times, t, side="right") - 1, len(self.times) - 2)
        span = self.times[index + 1] - self.times[index]
        delta = self.targets[index + 1] - self.targets[index]
        return self.targets[index] + (t - self.times[index]) / span * delta, delta / span * self.speed

    def control(self, model, data):
        self.reference, velocity = self.sample(data.time)
        # Velocity and bias-force feedforward through position actuators;
        # model force limits still apply, and actual motion is integrated by mj_step.
        data.ctrl[self.ids] = self.reference + (3 * velocity + data.qfrc_bias[self.dadr]) / 80

    def record(self, model, data):
        target, _ = self.sample(data.time)
        self.max_errors = np.maximum(self.max_errors, abs(data.qpos[self.qadr] - target))
        shoulder = data.body("RShoulder").xpos
        wrist = data.body("r_wrist").xpos
        self.max_height_error = max(self.max_height_error, abs(wrist[2] - shoulder[2]))
        # Wrist local Y is vertical for the requested +90 degree wrist pose.
        angle = np.arccos(np.clip(abs(data.body("r_wrist").xmat.reshape(3, 3)[2, 1]), 0, 1))
        self.max_wrist_vertical_error = max(self.max_wrist_vertical_error, angle)
        if not self.completed and data.time >= self.duration:
            self.completed = True
            print("CSV trajectory completed; holding endpoint" if not self.loop else "CSV trajectory completed; smooth return and repeat", flush=True)

    def report(self):
        errors = np.rad2deg(self.max_errors)
        print("Max joint error (deg): " + ", ".join(f"{n}={v:.3f}" for n, v in zip(self.names, errors)))
        print(f"Max shoulder-to-wrist height difference: {1000*self.max_height_error:.3f} mm")
        print(f"Max wrist vertical error: {np.rad2deg(self.max_wrist_vertical_error):.3f} deg")
        if max(errors) > 1 or self.max_height_error > 0.001 or self.max_wrist_vertical_error > np.deg2rad(1):
            raise RuntimeError("Trajectory tracking exceeded 1 deg / 1 mm tolerance")

"""Q1 lite live interactive viewer (opens a window, no GIF).

Run: /Users/dan/miniconda3/envs/fly/bin/python3 view_demo.py

- Opens a native MuJoCo window; drag to orbit, scroll to zoom, right-drag to pan.
- Robot walks with a spider trot gait (correct forward +Y) around a spread stance.
- Press Esc / close window to quit.

Leg conventions (see q1lite.xml):
  Right legs (FR/RR): local +X = world +X;  Left legs (FL/RL): frame rotated 180 deg
  about Z, local +X = world -X.  Positive local hip angle swings the foot toward the
  local +Y side, so LEFT legs need the oscillation sign flipped to mirror the right.

Adjust gait with the params below (edit and re-run).
"""
import time
import numpy as np
import mujoco
import mujoco.viewer

XML = "q1lite.xml"
LEGS = ["FL", "FR", "RL", "RR"]
# Standby stance (official firmware pose, model-local radians):
# FR +45deg, RR -45deg, FL -45deg, RL +45deg
HIP0 = {"FL": -0.785, "FR": +0.785, "RL": +0.785, "RR": -0.785}
# Trot: diagonal pairs FL+RR / FR+RL move together
PHASE = {"FL": 0.0, "FR": np.pi, "RL": np.pi, "RR": 0.0}
# Left-right mirror: left legs oscillate with opposite sign (+sin vs -sin)
SIDE = {"FL": -1.0, "RL": -1.0, "FR": +1.0, "RR": +1.0}

F, A_HIP, A_KNEE = 1.3, 0.4, 0.6   # gait params
GAIT = True                          # False -> just inspect standing pose


def make_ctrl(model, data, t):
    kp, kd = 5.0, 0.3
    ctrl = np.zeros(model.nu)
    for leg in LEGS:
        th = 2 * np.pi * F * t + PHASE[leg]
        q_hip = HIP0[leg] + SIDE[leg] * A_HIP * np.sin(th)
        q_knee = A_KNEE * (0.5 - 0.5 * np.cos(th))
        for j, qdes in (("hip", q_hip), ("knee", q_knee)):
            jid = model.joint(f"{leg}_{j}_joint")
            act = model.actuator(f"{leg}_{j}")
            ctrl[act.id] = kp * (qdes - data.qpos[jid.qposadr[0]]) - kd * data.qvel[jid.dofadr[0]]
    return ctrl


def main():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("stand").id)

    t0 = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            t = time.time() - t0
            data.ctrl[:] = make_ctrl(model, data, t) if GAIT else 0.0
            mujoco.mj_step(model, data)
            viewer.sync()
            el = time.time() - step_start
            if el < model.opt.timestep:
                time.sleep(model.opt.timestep - el)


if __name__ == "__main__":
    main()

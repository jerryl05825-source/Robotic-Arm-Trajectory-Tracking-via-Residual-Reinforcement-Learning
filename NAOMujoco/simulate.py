"""NAO V4 viewer and headless smoke check."""
import argparse
from pathlib import Path
import time

import mujoco
import numpy as np

from build_model import MODELS, build


def main():
    parser = argparse.ArgumentParser(description="NAO V4 MuJoCo simulation")
    parser.add_argument("--floating", action="store_true", help="Free base; no balance controller")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seconds", type=float, default=5.0, help="Headless simulation duration")
    parser.add_argument("--screenshot", type=Path, help="Save a PNG of the initial pose")
    parser.add_argument("--trajectory", type=Path, help="Right-arm joint trajectory CSV")
    parser.add_argument("--loop", action="store_true", help="Repeat trajectory with a smooth return")
    parser.add_argument("--speed", type=float, default=1.0, help="Trajectory playback speed")
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error("--speed must be positive and finite")
    if args.trajectory and args.floating:
        parser.error("Right-arm playback currently requires the fixed torso scene")
    path = MODELS / ("nao_v4_floating.xml" if args.floating else "nao_v4.xml")
    if args.rebuild or not path.exists():
        build(args.floating)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    trajectory = None
    if args.trajectory:
        from trajectory import RightArmTrajectory
        trajectory = RightArmTrajectory(args.trajectory, model, data, args.speed, args.loop)
        print(f"Playing {trajectory.path.name}: {trajectory.duration:.2f} s; wrist=90 deg, hand=closed", flush=True)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, 0, 0.29]
    camera.distance, camera.azimuth, camera.elevation = 1.3, (225 if trajectory else 135), -15
    if args.screenshot:
        from PIL import Image
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        with mujoco.Renderer(model, height=900, width=1200) as renderer:
            option = mujoco.MjvOption()
            option.geomgroup[3] = 0
            renderer.update_scene(data, camera=camera, scene_option=option)
            Image.fromarray(renderer.render()).save(args.screenshot)
        print(f"Screenshot: {args.screenshot.resolve()}")
    print(f"NAO V4 | {'floating base (no balance controller)' if args.floating else 'fixed torso'} | "
          f"nq={model.nq}, nv={model.nv}, actuators={model.nu}")
    if args.headless:
        duration = max(args.seconds, trajectory.duration + 1) if trajectory else args.seconds
        for _ in range(int(np.ceil(duration / model.opt.timestep))):
            if trajectory:
                trajectory.control(model, data)
            mujoco.mj_step(model, data)
            if trajectory:
                trajectory.record(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError("Non-finite simulation state")
        warnings = {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number}
        if warnings:
            raise RuntimeError(f"MuJoCo warnings: {warnings}")
        print(f"PASS: simulated {data.time:.3f} s; finite state, no MuJoCo warnings; contacts={data.ncon}")
        if trajectory:
            trajectory.report()
        return
    from mujoco import viewer as viewer_module
    with viewer_module.launch_passive(model, data) as viewer:
        viewer.opt.geomgroup[3] = 0
        viewer.cam.lookat[:] = camera.lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
        wall_start = time.monotonic()
        while viewer.is_running():
            start = time.monotonic()
            target_time = start - wall_start
            with viewer.lock():
                # Decouple the 500 Hz physics loop from viewer rendering.
                # Slow graphics must not stretch a four-second CSV into minutes.
                while data.time < target_time + model.opt.timestep:
                    if trajectory:
                        trajectory.control(model, data)
                    mujoco.mj_step(model, data)
                    if trajectory:
                        trajectory.record(model, data)
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.monotonic() - start)))


if __name__ == "__main__":
    main()

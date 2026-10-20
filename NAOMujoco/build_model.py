"""Build reproducible MuJoCo scenes from the supplied NAO V4 URDF/STL assets."""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from convert_meshes import convert

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "Nao V4"
MODELS = HERE / "models"


def save(root, path):
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def build(floating=False):
    MODELS.mkdir(exist_ok=True)
    urdf = ET.parse(SOURCE / "urdf/naoV40_generated_urdf/nao.urdf").getroot()
    joints = {j.get("name"): j for j in urdf.findall("joint") if j.get("type") == "revolute"}
    # Convert DAE visuals to OBJ; keep STL collision geometry and URDF scale.
    for element in urdf.findall("gazebo") + urdf.findall("transmission"):
        urdf.remove(element)
    for mesh in urdf.iter("mesh"):
        name = Path(mesh.get("filename")).name
        name = name.replace("LAnklePitch", "L_AnklePitch").replace("RAnklePitch", "R_AnklePitch")
        path = SOURCE / "meshes/V40" / name
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.suffix == ".dae":
            target = MODELS / "visual_meshes" / (path.stem + ".obj")
            convert(path, target)
            path = target
        mesh.set("filename", str(path))
    ET.SubElement(ET.SubElement(urdf, "mujoco"), "compiler",
                  discardvisual="false", fusestatic="false", strippath="false", balanceinertia="true")
    imported = mujoco.MjModel.from_xml_string(ET.tostring(urdf, encoding="unicode"))
    with tempfile.TemporaryDirectory() as tmp:
        converted = str(Path(tmp) / "converted.xml")
        mujoco.mj_saveLastXML(converted, imported)
        root = ET.parse(converted).getroot()
    root.set("model", "NAO V4 - " + ("floating base" if floating else "fixed torso"))
    for mesh in root.findall("asset/mesh"):
        original = Path(mesh.get("file"))
        mesh.set("file", ("visual_meshes/" if original.suffix == ".obj" else "../../Nao V4/meshes/V40/") + original.name)
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0.15", armature="0.001")
    ET.SubElement(default, "geom", friction="0.9 0.005 0.0001", condim="3")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1200", offheight="900")
    ET.SubElement(visual, "headlight", diffuse="0.6 0.6 0.6", ambient="0.3 0.3 0.3")
    ET.SubElement(root, "statistic", center="0 0 0.28", extent="0.85")
    asset = root.find("asset")
    ET.SubElement(asset, "texture", name="floor_tex", type="2d", builtin="checker",
                  rgb1="0.19 0.23 0.29", rgb2="0.25 0.30 0.37", width="512", height="512")
    ET.SubElement(asset, "material", name="floor_mat", texture="floor_tex", texrepeat="8 8", reflectance="0.1")
    world = root.find("worldbody")
    torso = world.find(".//body[@name='torso']")
    if floating:
        base = world.find("body[@name='base_link']")
        base.remove(torso)
        world.remove(base)
        world.append(torso)
        torso.append(base)
        ET.SubElement(torso, "freejoint", name="root")
    for geom in world.iter("geom"):
        geom.set("rgba", "0.86 0.89 0.92 1")
        # Ground contacts only in this first integration; convex link hulls
        # otherwise create spurious self contacts around the fingers/joints.
        is_visual = geom.get("contype") == "0" and geom.get("conaffinity") == "0"
        geom.set("group", "2" if is_visual else "3")
        geom.set("contype", "0" if is_visual else "2")
        geom.set("conaffinity", "0" if is_visual else "1")
        if any(s in geom.get("mesh", "") for s in ("ShoulderRoll", "HipYawPitch", "HeadYaw")):
            geom.set("rgba", "0.12 0.40 0.55 1")
    ET.SubElement(world, "geom", name="floor", type="plane", size="2 2 0.05",
                  material="floor_mat", contype="1", conaffinity="2")
    ET.SubElement(world, "light", pos="1 -1 2", dir="-0.4 0.4 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "camera", name="overview", pos="1.0 -1.0 0.8", xyaxes="0.707 0.707 0 -0.3 0.3 0.9")
    equality = ET.SubElement(root, "equality")
    actuator = ET.SubElement(root, "actuator")
    for name, joint in joints.items():
        mimic = joint.find("mimic")
        if mimic is not None:
            ET.SubElement(equality, "joint", name=name + "_coupling", joint1=name,
                          joint2=mimic.get("joint"),
                          polycoef=f"{mimic.get('offset', '0')} {mimic.get('multiplier', '1')} 0 0 0")
        else:
            limits = joint.find("limit")
            effort = limits.get("effort")
            ET.SubElement(actuator, "position", name=name, joint=name, kp="30", kv="2",
                          ctrllimited="true", ctrlrange=f"{limits.get('lower')} {limits.get('upper')}",
                          forcelimited="true", forcerange=f"-{effort} {effort}")
    path = MODELS / ("nao_v4_floating.xml" if floating else "nao_v4.xml")
    save(root, path)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    pose = {"LShoulderPitch": 1.4, "RShoulderPitch": 1.4,
            "LShoulderRoll": 0.15, "RShoulderRoll": -0.15,
            "LElbowRoll": -0.5, "RElbowRoll": 0.5, "LHand": 0.2, "RHand": 0.2}
    for name, joint in joints.items():
        mimic = joint.find("mimic")
        value = pose.get(name, 0.0)
        if mimic is not None:
            value = pose.get(mimic.get("joint"), 0.0) * float(mimic.get("multiplier", 1)) + float(mimic.get("offset", 0))
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    # Put the lowest mesh vertex at the floor, respecting scale and mesh frames.
    bottom = np.inf
    for g in range(model.ngeom):
        mesh_id = model.geom_dataid[g]
        if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        start, count = model.mesh_vertadr[mesh_id], model.mesh_vertnum[mesh_id]
        vertices = model.mesh_vert[start:start + count]
        heights = vertices @ data.geom_xmat[g].reshape(3, 3)[2] + data.geom_xpos[g, 2]
        bottom = min(bottom, float(heights.min()))
    height = -bottom + 0.0005
    torso.set("pos", f"0 0 {height:.9f}")
    if floating:
        data.qpos[model.joint("root").qposadr[0] + 2] = height
    for a in range(model.nu):
        data.ctrl[a] = data.qpos[model.jnt_qposadr[model.actuator_trnid[a, 0]]]
    ET.SubElement(ET.SubElement(root, "keyframe"), "key", name="home",
                  qpos=" ".join(map(str, data.qpos)), ctrl=" ".join(map(str, data.ctrl)))
    save(root, path)
    checked = mujoco.MjModel.from_xml_path(str(path))
    print(f"Built {path.name}: {checked.njnt} joints, {checked.nu} actuators, "
          f"{checked.neq} couplings, mass={checked.body_mass.sum():.3f} kg")
    return path


if __name__ == "__main__":
    build()
    build(floating=True)

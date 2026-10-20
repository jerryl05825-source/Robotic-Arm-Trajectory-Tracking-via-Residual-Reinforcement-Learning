"""Convert this asset set's COLLADA geometry to OBJ without extra packages.

Preserves scene matrices and triangulates polylist faces. Texture/material
conversion is deliberately omitted; the scene uses a uniform robot material.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}


def convert(source: Path, target: Path):
    root = ET.parse(source).getroot()
    geometries = {g.get("id"): g for g in root.findall(".//c:library_geometries/c:geometry", NS)}
    lines, offset = [], 0
    for node in root.findall(".//c:visual_scene/c:node", NS):
        if node.find("c:node", NS) is not None:
            raise ValueError(f"Unexpected nested COLLADA node: {source}")
        matrix = node.find("c:matrix", NS)
        transform = np.eye(4) if matrix is None else np.fromstring(matrix.text, sep=" ").reshape(4, 4)
        for instance in node.findall("c:instance_geometry", NS):
            mesh = geometries[instance.get("url")[1:]].find("c:mesh", NS)
            inputs = mesh.find("c:vertices", NS)
            position_id = next(i.get("source")[1:] for i in inputs if i.get("semantic") == "POSITION")
            source_element = mesh.find(f"c:source[@id='{position_id}']", NS)
            positions = np.fromstring(source_element.find("c:float_array", NS).text, sep=" ").reshape(-1, 3)
            positions = positions @ transform[:3, :3].T + transform[:3, 3]
            lines.extend("v " + " ".join(f"{v:.9g}" for v in p) for p in positions)
            for primitive in list(mesh):
                kind = primitive.tag.split("}")[-1]
                if kind not in ("polylist", "triangles"):
                    continue
                input_elements = primitive.findall("c:input", NS)
                stride = max(int(i.get("offset", 0)) for i in input_elements) + 1
                vertex_offset = next(int(i.get("offset", 0)) for i in input_elements if i.get("semantic") == "VERTEX")
                indices = np.fromstring(primitive.find("c:p", NS).text, sep=" ", dtype=int).reshape(-1, stride)[:, vertex_offset]
                counts = ([3] * int(primitive.get("count")) if kind == "triangles" else
                          np.fromstring(primitive.find("c:vcount", NS).text, sep=" ", dtype=int))
                cursor = 0
                for count in counts:
                    face = indices[cursor:cursor + count] + offset + 1
                    for k in range(1, count - 1):
                        lines.append(f"f {face[0]} {face[k]} {face[k + 1]}")
                    cursor += count
            offset += len(positions)
    if not offset:
        raise ValueError(f"No scene geometry in {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")

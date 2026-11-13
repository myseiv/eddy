"""Forward and inverse kinematics for any serial arm.

A Chain is a list of joints, each with a fixed offset from the previous joint and
an axis it turns about (or slides along). Chains can be built by hand or read from
a URDF, which is how Eddy learns the geometry of a robot it has never seen.

Inverse kinematics is numerical (damped least squares with random restarts), so it
needs no per-robot maths and works for any number of joints.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) or 1.0)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def transform(xyz=(0, 0, 0), rpy=(0, 0, 0)) -> np.ndarray:
    """4x4 transform from a translation and URDF-style roll/pitch/yaw."""
    r, p, y = rpy
    R = _rot(np.array([0, 0, 1.0]), y) @ _rot(np.array([0, 1.0, 0]), p) @ _rot(np.array([1.0, 0, 0]), r)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


@dataclass
class Joint:
    name: str
    kind: str = "revolute"            # revolute, continuous, prismatic, fixed
    origin: np.ndarray = None         # 4x4 offset from the previous joint's frame
    axis: Sequence[float] = (0, 0, 1)
    lower: float = -math.pi
    upper: float = math.pi
    max_velocity: float = 1.5

    def __post_init__(self):
        if self.origin is None:
            self.origin = np.eye(4)
        self.axis = np.asarray(self.axis, dtype=float)
        if self.kind == "continuous":
            self.lower, self.upper = -math.pi, math.pi

    @property
    def moves(self) -> bool:
        return self.kind in ("revolute", "continuous", "prismatic")


@dataclass
class IKResult:
    success: bool
    q: List[float]
    error: float          # metres between the tool and the target
    iterations: int

    def as_dict(self, chain: "Chain") -> Dict[str, float]:
        return dict(zip(chain.joint_names, self.q))


class Chain:
    def __init__(self, joints: List[Joint], name: str = "arm"):
        self.name = name
        self.joints = joints
        self.active = [j for j in joints if j.moves]

    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self.active]

    @property
    def dof(self) -> int:
        return len(self.active)

    def _frames(self, q: Sequence[float]):
        """Tool transform plus (axis, position, kind) of each moving joint, in the base frame."""
        T = np.eye(4)
        info = []
        i = 0
        for j in self.joints:
            T = T @ j.origin
            if not j.moves:
                continue
            axis_w = T[:3, :3] @ j.axis
            info.append((axis_w / (np.linalg.norm(axis_w) or 1.0), T[:3, 3].copy(), j.kind))
            M = np.eye(4)
            if j.kind == "prismatic":
                M[:3, 3] = j.axis / (np.linalg.norm(j.axis) or 1.0) * q[i]
            else:
                M[:3, :3] = _rot(j.axis, q[i])
            T = T @ M
            i += 1
        return T, info

    def fk(self, q: Sequence[float]) -> np.ndarray:
        if len(q) != self.dof:
            raise ValueError(f"expected {self.dof} joint values, got {len(q)}")
        return self._frames(q)[0]

    def position(self, q: Sequence[float]) -> np.ndarray:
        return self.fk(q)[:3, 3]

    def jacobian(self, q: Sequence[float]) -> np.ndarray:
        T, info = self._frames(q)
        tool = T[:3, 3]
        J = np.zeros((3, self.dof))
        for i, (axis, pos, kind) in enumerate(info):
            J[:, i] = axis if kind == "prismatic" else np.cross(axis, tool - pos)
        return J

    def clamp(self, q: Sequence[float]) -> np.ndarray:
        lo = np.array([j.lower for j in self.active])
        hi = np.array([j.upper for j in self.active])
        return np.minimum(np.maximum(np.asarray(q, dtype=float), lo), hi)

    def reach(self) -> float:
        """Upper bound on how far the tool can get from the base."""
        total = 0.0
        for j in self.joints:
            total += float(np.linalg.norm(j.origin[:3, 3]))
            if j.kind == "prismatic":
                total += max(abs(j.lower), abs(j.upper))
        return total

    def ik(self, target: Sequence[float], q0: Optional[Sequence[float]] = None,
           tol: float = 1e-3, max_iter: int = 300, restarts: int = 12, seed: int = 0) -> IKResult:
        """Find joint values that put the tool at `target` (x, y, z in the base frame)."""
        target = np.asarray(target, dtype=float)
        if self.dof == 0:
            return IKResult(False, [], float("inf"), 0)
        rng = np.random.default_rng(seed)
        lo = np.array([j.lower for j in self.active])
        hi = np.array([j.upper for j in self.active])
        start = self.clamp(q0 if q0 is not None else np.zeros(self.dof))
        best = IKResult(False, list(start), float(np.linalg.norm(target - self.position(start))), 0)
        total_iters = 0
        for attempt in range(restarts + 1):
            q = start.copy() if attempt == 0 else rng.uniform(lo, hi)
            damping = 0.05
            for _ in range(max_iter):
                total_iters += 1
                err = target - self.position(q)
                dist = float(np.linalg.norm(err))
                if dist < tol:
                    break
                J = self.jacobian(q)
                dq = J.T @ np.linalg.solve(J @ J.T + (damping ** 2) * np.eye(3), err)
                step = float(np.max(np.abs(dq)))
                if step > 0.3:
                    dq *= 0.3 / step
                q = np.minimum(np.maximum(q + dq, lo), hi)
            dist = float(np.linalg.norm(target - self.position(q)))
            if dist < best.error:
                best = IKResult(dist < tol, [float(v) for v in q], dist, total_iters)
            if best.success:
                break
        best.iterations = total_iters
        return best

    def to_urdf(self) -> str:
        """Write the chain back out as URDF, so one Eddy can hand its arm geometry to another."""
        links = [f'  <link name="link0"/>']
        joints = []
        for i, j in enumerate(self.joints):
            R, t = j.origin[:3, :3], j.origin[:3, 3]
            rpy = (math.atan2(R[2, 1], R[2, 2]), math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])),
                   math.atan2(R[1, 0], R[0, 0]))
            links.append(f'  <link name="link{i + 1}"/>')
            joints.append(
                f'  <joint name="{j.name}" type="{j.kind}">\n'
                f'    <parent link="link{i}"/><child link="link{i + 1}"/>\n'
                f'    <origin xyz="{t[0]:.6f} {t[1]:.6f} {t[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"/>\n'
                f'    <axis xyz="{j.axis[0]:.6f} {j.axis[1]:.6f} {j.axis[2]:.6f}"/>\n'
                f'    <limit lower="{j.lower:.6f}" upper="{j.upper:.6f}" velocity="{j.max_velocity:.3f}" effort="1"/>\n'
                f'  </joint>')
        return f'<robot name="{self.name}">\n' + "\n".join(links + joints) + "\n</robot>"

    # ---- construction -------------------------------------------------------

    @classmethod
    def from_urdf(cls, xml_text: str, base: Optional[str] = None, tip: Optional[str] = None,
                  allowed: Optional[Sequence[str]] = None) -> "Chain":
        """Build a chain from URDF text. By default: from the root link to the deepest leaf
        that has the most moving, limited joints on the way (i.e. the arm, not a wheel).
        allowed: if given, only these joint names may move in the chain."""
        root = ET.fromstring(xml_text)
        by_child: Dict[str, dict] = {}
        children: Dict[str, List[str]] = {}
        for el in root.findall("joint"):
            parent = el.find("parent").get("link")
            child = el.find("child").get("link")
            origin = el.find("origin")
            xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
            rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
            axis_el = el.find("axis")
            axis = [float(v) for v in (axis_el.get("xyz", "1 0 0") if axis_el is not None else "1 0 0").split()]
            limit = el.find("limit")
            kind = el.get("type", "fixed")
            if kind not in ("revolute", "continuous", "prismatic"):
                kind = "fixed"
            j = Joint(
                name=el.get("name"), kind=kind, origin=transform(xyz, rpy), axis=axis,
                lower=float(limit.get("lower", -math.pi)) if limit is not None else -math.pi,
                upper=float(limit.get("upper", math.pi)) if limit is not None else math.pi,
                max_velocity=float(limit.get("velocity", 1.5)) if limit is not None else 1.5,
            )
            by_child[child] = {"joint": j, "parent": parent}
            children.setdefault(parent, []).append(child)
        if not by_child:
            raise ValueError("URDF has no joints")

        def path_to(link: str) -> List[Joint]:
            out = []
            while link in by_child and link != base:
                out.append(by_child[link]["joint"])
                link = by_child[link]["parent"]
            if base is not None and link != base:
                raise ValueError(f"link '{base}' is not an ancestor of '{tip}'")
            return list(reversed(out))

        if tip is None:
            def moving(link):
                return [j for j in path_to(link) if j.moves]

            if allowed is not None:
                # Furthest link reachable using only joints we are able to command
                # (so a gripper finger or a wheel past the arm is left out).
                ends = [c for c in by_child if moving(c) and all(j.name in allowed for j in moving(c))]
                if not ends:
                    raise ValueError("no chain can be built from the allowed joints")
            else:
                ends = [c for c in by_child if c not in children]

            def score(link):
                return (sum(1 for j in moving(link) if j.kind != "continuous"), len(path_to(link)))
            tip = max(ends, key=score)
        elif tip not in by_child:
            raise ValueError(f"URDF has no link '{tip}'")
        return cls(path_to(tip), name=root.get("name", "arm"))


def demo_arm() -> Chain:
    """A small 4-joint arm (turntable, shoulder, elbow, wrist) used by the simulator."""
    return Chain([
        Joint("base_yaw", "revolute", transform((0, 0, 0.10)), (0, 0, 1), -math.pi, math.pi),
        Joint("shoulder", "revolute", transform((0, 0, 0.05)), (0, 1, 0), -1.9, 1.9),
        Joint("elbow", "revolute", transform((0, 0, 0.20)), (0, 1, 0), -2.4, 2.4),
        Joint("wrist", "revolute", transform((0, 0, 0.20)), (0, 1, 0), -1.9, 1.9),
        Joint("tool", "fixed", transform((0, 0, 0.08))),
    ], name="demo_arm")

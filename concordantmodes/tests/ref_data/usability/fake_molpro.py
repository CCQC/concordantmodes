#!/usr/bin/env python3
"""Fake Molpro for testing: energy of water from a model potential.
Reads input.dat (xyz block inside 'geometry = {'), writes Molpro-like output."""

import sys, math, re, os

args = sys.argv[1:]
inp = "input.dat"
out = "output.dat"
if "-o" in args:
    out = args[args.index("-o") + 1]
text = open(inp).read()
level_A = "cc-pvtz" in text.lower()
block = (
    re.split(r"geometry\s*=\s*\{", text, maxsplit=1)[1]
    .split("}", 1)[0]
    .strip()
    .splitlines()
)
atoms = [l.split() for l in block if len(l.split()) == 4]
xyz = [[float(v) for v in a[1:]] for a in atoms]


def d(a, b):
    return math.dist(xyz[a], xyz[b])


def ang(a, b, c):
    v1 = [xyz[a][i] - xyz[b][i] for i in range(3)]
    v2 = [xyz[c][i] - xyz[b][i] for i in range(3)]
    dot = sum(p * q for p, q in zip(v1, v2))
    return math.acos(dot / (math.hypot(*v1) * math.hypot(*v2)))


r1, r2, th = d(0, 1), d(2, 1), ang(0, 1, 2)
r0, th0 = 1.8100, math.radians(104.5)
kr, ka, krr, kra = (
    (0.52, 0.16, -0.010, 0.020) if level_A else (0.55, 0.17, -0.012, 0.025)
)
dr1, dr2, dth = r1 - r0, r2 - r0, th - th0
E = (
    -76.3
    + 0.5 * kr * (dr1**2 + dr2**2)
    + 0.5 * ka * dth**2
    + krr * dr1 * dr2
    + kra * (dr1 + dr2) * dth
    - 0.3 * (dr1**3 + dr2**3)
)
with open(out, "w") as f:
    f.write(" !RHF STATE 1.1 Energy               %.12f\n" % (E + 0.3))
    f.write(" !CCSD(T) total energy               %.12f\n" % E)
    f.write(" CCSD(T)/cc-pVDZ energy=   %.12f\n\n Molpro calculation terminated\n" % E)

# ---- Parameters (from the project report) ----
PRICE = 1000
OPM = 0.15
RHO = 0.5
SUPPORT_COST = 5000
LAMBDA = 0.95
TOL = 1e-6

UNITS = {"H": 500, "M": 250, "L": 20}
OBSOLESCENCE = {"S": 10000, "M": 40000, "L": 80000}

DEMANDS = ["H", "M", "L"]
GAPS = ["S", "M", "L"]
ACTIONS = ["Maintain", "PhaseOut", "Delete"]

# Technology gap transition: P(G' | G)
GAP_TRANS = {
    "S": [0.95, 0.05, 0.00],
    "M": [0.00, 0.90, 0.10],
    "L": [0.00, 0.00, 1.00],
}

# Demand transition: P(D' | D, G, action). Rows = current demand H, M, L.
DEMAND_TRANS = {
    ("Maintain", "S"): [[0.75, 0.25, 0.00], [0.10, 0.70, 0.20], [0.00, 0.20, 0.80]],
    ("Maintain", "M"): [[0.55, 0.40, 0.05], [0.05, 0.60, 0.35], [0.00, 0.10, 0.90]],
    ("Maintain", "L"): [[0.30, 0.50, 0.20], [0.00, 0.40, 0.60], [0.00, 0.05, 0.95]],
    ("PhaseOut", "S"): [[0.50, 0.45, 0.05], [0.00, 0.65, 0.35], [0.00, 0.10, 0.90]],
    ("PhaseOut", "M"): [[0.35, 0.50, 0.15], [0.00, 0.50, 0.50], [0.00, 0.05, 0.95]],
    ("PhaseOut", "L"): [[0.10, 0.50, 0.40], [0.00, 0.25, 0.75], [0.00, 0.00, 1.00]],
}

STATES = [(d, g) for d in DEMANDS for g in GAPS]


def reward(d, g, action):
    if action == "Maintain":
        return UNITS[d] * PRICE * OPM - SUPPORT_COST - OBSOLESCENCE[g]
    if action == "PhaseOut":
        return RHO * UNITS[d] * PRICE * OPM
    return 0.0  # Delete


def q_value(d, g, action, V):
    r = reward(d, g, action)
    if action == "Delete":
        return r  # next state is Dead, value 0
    d_probs = DEMAND_TRANS[(action, g)][DEMANDS.index(d)]
    future = 0.0
    for i, d2 in enumerate(DEMANDS):
        for j, g2 in enumerate(GAPS):
            future += d_probs[i] * GAP_TRANS[g][j] * V[(d2, g2)]
    return r + LAMBDA * future


def value_iteration():
    V = {s: 0.0 for s in STATES}
    while True:
        V_new = {}
        for (d, g) in STATES:
            V_new[(d, g)] = max(q_value(d, g, a, V) for a in ACTIONS)
        diff = max(abs(V_new[s] - V[s]) for s in STATES)
        V = V_new
        if diff < TOL:
            break
    policy = {}
    for (d, g) in STATES:
        qs = {a: q_value(d, g, a, V) for a in ACTIONS}
        policy[(d, g)] = max(qs, key=qs.get)
    return V, policy


# Solve once when the module loads
V_STAR, POLICY = value_iteration()


def get_policy(demand, gap):
    """demand: 'H','M','L'; gap: 'S','M','L'. Returns the optimal action."""
    return POLICY[(demand, gap)]


if __name__ == "__main__":
    print("Optimal policy:")
    for (d, g) in STATES:
        print(f"{d}{g}: {POLICY[(d, g)]:10s}  V = {V_STAR[(d, g)]:,.0f}")

"""MDVRP reoptimization. Python 3.10+, standard library only.

Run: python mdvrp.py
Vertices are integer indices of w. Each entry in depots denotes one
vehicle's fixed depot. Clients must not include depot vertices.
A missing arc is represented by math.inf. Unused vehicles have route [].
Exact costs are certified; replan() returns a feasible heuristic result.
"""
from functools import lru_cache
from itertools import combinations
from math import inf, isfinite
from time import monotonic


def route_cost(w, depot, route):
    if not route:
        return 0
    seq = [depot, *route, depot]
    return sum(w[a][b] for a, b in zip(seq, seq[1:]))


def validate(w, depots, clients, capacity, routes):
    if len(routes) != len(depots):
        raise ValueError('One route per vehicle is required')
    flat = [i for r in routes for i in r]
    if sorted(flat) != sorted(clients) or len(set(flat)) != len(flat):
        raise ValueError('Each active client must occur exactly once')
    if any(len(r) > capacity for r in routes):
        raise ValueError('Capacity exceeded')
    cost = sum(route_cost(w, d, r) for d, r in zip(depots, routes))
    if not isfinite(cost):
        raise ValueError('Route uses a missing arc')
    return cost


class ExactDP:
    """Subset DP with route reconstruction and optional deadline.

    precompute() caches all active subsets for fast later solve() calls.
    For one query, only necessary states are evaluated.
    """
    def __init__(self, w, depots, clients, capacity, deadline=inf):
        self.w = w
        self.depots = tuple(depots)
        self.clients = tuple(clients)
        self.M = capacity
        self.deadline = deadline
        if capacity < 0 or len(set(clients)) != len(clients):
            raise ValueError('Invalid capacity or repeated clients')
        if set(clients) & set(depots):
            raise ValueError('Clients and depots must be disjoint')
        if any(len(row) != len(w) for row in w):
            raise ValueError('Expected square weight matrix')
        if any(x < 0 or x != x for row in w for x in row):
            raise ValueError('Weights must be nonnegative, not NaN')
        if any(i < 0 or i >= len(w) for i in (*clients, *depots)):
            raise ValueError('Vertex index outside weight matrix')

        def tick():
            if monotonic() >= self.deadline:
                raise TimeoutError('Exact subproblem time limit')

        @lru_cache(None)
        def path(d, mask, j):
            tick()
            rest = mask ^ (1 << j)
            if not rest:
                return w[d][self.clients[j]], -1
            return min((path(d, rest, i)[0]
                        + w[self.clients[i]][self.clients[j]], i)
                       for i in self.bits(rest))

        @lru_cache(None)
        def tour(d, mask):
            tick()
            if not mask:
                return 0, -1
            if mask.bit_count() > capacity:
                return inf, -1
            return min((path(d, mask, j)[0] + w[self.clients[j]][d], j)
                       for j in self.bits(mask))

        @lru_cache(None)
        def fleet(t, mask):
            tick()
            if not mask:
                return 0, 0
            if t == 0 or mask.bit_count() > t * capacity:
                return inf, 0
            best = (inf, 0)
            subset = mask
            while True:
                tick()
                if subset.bit_count() <= capacity:
                    value = (fleet(t-1, mask ^ subset)[0]
                             + tour(self.depots[t-1], subset)[0])
                    if value < best[0]:
                        best = value, subset
                if subset == 0:
                    break
                subset = (subset - 1) & mask
            return best

        self.path, self.tour, self.fleet = path, tour, fleet

    @staticmethod
    def bits(mask):
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    def precompute(self):
        for mask in range(1 << len(self.clients)):
            self.fleet(len(self.depots), mask)
        return self

    def solve(self, active=None):
        active = set(self.clients if active is None else active)
        if not active <= set(self.clients):
            raise ValueError('Unknown active client')
        mask = sum(1 << j for j, i in enumerate(self.clients) if i in active)
        value = self.fleet(len(self.depots), mask)[0]
        if not isfinite(value):
            return inf, None
        routes = [[] for _ in self.depots]
        for t in range(len(self.depots), 0, -1):
            subset = self.fleet(t, mask)[1]
            mask ^= subset
            j = self.tour(self.depots[t-1], subset)[1]
            backward = []
            while subset:
                backward.append(self.clients[j])
                previous = self.path(self.depots[t-1], subset, j)[1]
                subset ^= 1 << j
                j = previous
            routes[t-1] = backward[::-1]
        validate(self.w, self.depots, active, self.M, routes)
        return value, routes


def replan(w, depots, clients, old_routes, cancelled, capacity,
           seconds=1.0, max_group=3, max_clients=12):
    """Delete cancellations, then exactly optimize small vehicle groups.

    Returns (cost, routes). No global optimality claim. Time is a soft
    wall-clock budget: initialization and output are necessarily linear.
    If deletion breaks feasibility, attempts full exact feasibility repair
    within the remaining budget; may raise TimeoutError or ValueError.
    """
    deadline = monotonic() + max(0.0, seconds)
    validate(w, depots, clients, capacity, old_routes)
    cancelled = set(cancelled)
    if not cancelled <= set(clients):
        raise ValueError('Unknown cancelled client')
    active = set(clients) - cancelled
    routes = [[i for i in r if i in active] for r in old_routes]
    try:
        cost = validate(w, depots, active, capacity, routes)
    except ValueError:
        cost, routes = ExactDP(w, depots, sorted(active), capacity,
                               deadline).solve()
        if routes is None:
            raise ValueError('Updated instance is infeasible')
        return cost, routes
    affected = {k for k, r in enumerate(old_routes) if cancelled & set(r)}
    limit = min(max_group, len(depots))
    changed = True
    while changed and monotonic() < deadline:
        changed = False
        for priority in (True, False):
            for size in range(1, limit+1):
                for group in combinations(range(len(depots)), size):
                    if monotonic() >= deadline:
                        return cost, routes
                    if bool(affected & set(group)) != priority:
                        continue
                    selected = [i for k in group for i in routes[k]]
                    if not selected or len(selected) > max_clients:
                        continue
                    ds = [depots[k] for k in group]
                    old = sum(route_cost(w, depots[k], routes[k])
                              for k in group)
                    try:
                        value, replacement = ExactDP(
                            w, ds, selected, capacity, deadline).solve()
                    except TimeoutError:
                        return cost, routes
                    if value < old:
                        for k, r in zip(group, replacement):
                            routes[k] = r
                        cost = validate(w, depots, active, capacity, routes)
                        changed = True
    return cost, routes


def qubo(w, depots, clients, capacity, feasible_upper_bound):
    """Return (names, Q, constant), energy = constant + sum Q[a,b]*v[a]*v[b].

    Q stores each upper-triangular pair once (no extra factor of two).
    Caller supplies a finite, nonnegative feasible upper bound.
    Missing arcs incur an additional penalty. This constructs a model;
    it does not assert that an annealer finds its global minimum.
    """
    if not isfinite(feasible_upper_bound) or feasible_upper_bound < 0:
        raise ValueError('Expected finite feasible upper bound')
    P = min(capacity, len(clients))
    if clients and P < 1:
        raise ValueError('Positive capacity required for nonempty clients')
    names, x, z = [], {}, {}
    for k in range(len(depots)):
        for p in range(P):
            for i in clients:
                x[i, p, k] = len(names)
                names.append(('x', i, p, k))
            z[p, k] = len(names)
            names.append(('z', p, k))
    Q, constant = {}, 0
    penalty = feasible_upper_bound + max(1, feasible_upper_bound)

    def add(a, b, coefficient):
        key = min(a, b), max(a, b)
        Q[key] = Q.get(key, 0) + coefficient

    def square(terms, target):
        nonlocal constant
        constant += penalty * target * target
        for u, (a, ca) in enumerate(terms):
            add(a, a, penalty * (ca * ca - 2 * target * ca))
            for b, cb in terms[u+1:]:
                add(a, b, 2 * penalty * ca * cb)

    def weight(a, b):
        return w[a][b] if isfinite(w[a][b]) else penalty

    for i in clients:
        square([(x[i, p, k], 1) for k in range(len(depots))
                for p in range(P)], 1)
    for k, d in enumerate(depots):
        for p in range(P):
            square([(x[i, p, k], 1) for i in clients] + [(z[p, k], -1)], 0)
            if p+1 < P:
                add(z[p+1, k], z[p+1, k], penalty)
                add(z[p+1, k], z[p, k], -penalty)
            for i in clients:
                a = x[i, p, k]
                if p == 0:
                    add(a, a, weight(d, i))
                add(a, a, weight(i, d))
                if p+1 < P:
                    add(a, z[p+1, k], -weight(i, d))
                    for j in clients:
                        add(a, x[j, p+1, k], weight(i, j))
    return names, {key: v for key, v in Q.items() if v}, constant


def demo():
    # Depot 0; clients a=1, b=2, c=3. Delete b.
    w = [[0 if i == j else 1 for j in range(4)] for i in range(4)]
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        w[a][b] = 0
    w[1][3] = 100
    old = [[1, 2, 3]]
    initial = [[1, 3]]
    print('Old optimum:', validate(w, [0], [1, 2, 3], 3, old))
    print('Deletion only:', validate(w, [0], [1, 3], 3, initial))
    value, routes = replan(w, [0], [1, 2, 3], old, {2}, 3)
    print('Replanned:', value, routes)
    cache = ExactDP(w, [0], [1, 2, 3], 3).precompute()
    print('Exact cached query:', cache.solve({1, 3}))
    names, Q, constant = qubo(w, [0], [1, 3], 3, 100)
    print('QUBO variables / terms / constant:', len(names), len(Q), constant)


if __name__ == '__main__':
    demo()

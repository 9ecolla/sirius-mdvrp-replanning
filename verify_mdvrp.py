"""Independent checks for mdvrp.py. Run: python verify_mdvrp.py"""
from itertools import permutations, product
from math import inf, isfinite
from random import Random
from mdvrp import ExactDP, route_cost, validate, replan, qubo


def brute(w, depots, clients, M):
    best = inf
    for owners in product(range(len(depots)), repeat=len(clients)):
        groups = [tuple(i for i, k in zip(clients, owners) if k == v)
                  for v in range(len(depots))]
        if any(len(g) > M for g in groups):
            continue
        value = sum(min(route_cost(w, d, p) for p in permutations(g))
                    for d, g in zip(depots, groups))
        best = min(best, value)
    return best


def check_dp_and_replan():
    rng = Random(17)
    queries = updates = 0
    for case in range(30):
        n, K, A = rng.randint(1, 5), rng.randint(1, 2), rng.randint(1, 2)
        M = rng.randint(1, n)
        w = [[0 if i == j else
              (inf if case % 3 == 0 and rng.random() < .15 else rng.randint(0, 20))
              for j in range(n+K)] for i in range(n+K)]
        depots = [n+k for k in range(K) for _ in range(A)]
        dp = ExactDP(w, depots, list(range(n)), M).precompute()
        _, old = dp.solve()
        for mask in range(1 << n):
            clients = [i for i in range(n) if mask >> i & 1]
            expected = brute(w, depots, clients, M)
            value, routes = dp.solve(clients)
            assert value == expected, (case, mask, value, expected)
            if routes is not None:
                assert validate(w, depots, clients, M, routes) == value
            queries += 1
            if old is not None and all(isfinite(x) for row in w for x in row):
                cancelled = set(range(n)) - set(clients)
                initial = [[i for i in r if i in clients] for r in old]
                upper = validate(w, depots, clients, M, initial)
                cost, updated = replan(w, depots, list(range(n)), old,
                                      cancelled, M, seconds=2,
                                      max_group=len(depots), max_clients=n)
                assert validate(w, depots, clients, M, updated) == cost
                assert expected <= cost <= upper
                # A group containing the whole fleet makes this small run exact.
                assert cost == expected
                updates += 1
    return queries, updates


def check_qubo():
    rng = Random(19)
    checked = 0
    for n, q, P in [(2, 1, 2), (2, 2, 2), (3, 2, 2)]:
        for sparse in [False, True]:
            w = [[0 if i == j else rng.randint(0, 12)
                  for j in range(n+q)] for i in range(n+q)]
            if sparse:
                w[n][0] = inf
                w[0][1] = inf
            clients, depots = list(range(n)), list(range(n, n+q))
            optimum = brute(w, depots, clients, P)
            if not isfinite(optimum):
                continue
            names, Q, constant = qubo(w, depots, clients, P, optimum)
            best = inf
            for bits in product((0, 1), repeat=len(names)):
                energy = constant + sum(c * bits[a] * bits[b]
                                        for (a, b), c in Q.items())
                values = dict(zip(names, bits))
                feasible = all(sum(values['x', i, p, k]
                                   for k in range(q) for p in range(P)) == 1
                               for i in clients)
                feasible &= all(sum(values['x', i, p, k] for i in clients)
                                == values['z', p, k]
                                for k in range(q) for p in range(P))
                feasible &= all(values['z', p+1, k] <= values['z', p, k]
                                for k in range(q) for p in range(P-1))
                if feasible:
                    routes = [[i for p in range(P) for i in clients
                               if values['x', i, p, k]] for k in range(q)]
                    cost = sum(route_cost(w, d, r) for d, r in zip(depots, routes))
                    feasible &= isfinite(cost)
                    if feasible:
                        assert energy == cost
                if not feasible:
                    assert energy > optimum
                best = min(best, energy)
                checked += 1
            assert best == optimum
    return checked


if __name__ == '__main__':
    queries, updates = check_dp_and_replan()
    print('DP queries checked:', queries)
    print('Replanning queries checked:', updates)
    print('QUBO assignments checked:', check_qubo())
    print('All checks passed')

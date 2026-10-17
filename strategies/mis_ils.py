"""Fixed baseline strategy: iterated local search for maximum independent sets.

Follows Andrade, Resende and Werneck, "Fast local search for the maximum
independent set problem" (J. Heuristics, 2012): (1,2)-swaps to a local optimum,
forced insertions as perturbation, and their acceptance rule.

The graph supplies num_vertices, neighbors(v) and adjacent(u, w); the rng
supplies randbelow(n) and random(). Results depend only on those inputs.

Tunable settings (defaults reproduce the original fixed strategy exactly):
max_perturbation and perturb_continue_pct shape the geometric number of forced
vertices, perturb_radius bounds how far extra forced vertices lie from the
first, tournament picks the oldest of that many outside vertices, and
accept_scale_pct scales the ARW probability of accepting a worse solution.
`work` counts neighbor-list entries scanned, a deterministic measure of compute.
"""

from array import array


class IteratedLocalSearch:
    def __init__(self, graph, rng, initial=(), max_perturbation=4, tournament=4,
                 perturb_continue_pct=50, perturb_radius=2, accept_scale_pct=100):
        n = graph.num_vertices
        self.graph = graph
        self.rng = rng
        self.max_perturbation = max_perturbation
        self.tournament = tournament
        self.continue_probability = perturb_continue_pct / 100
        self.perturb_radius = perturb_radius
        self.accept_scale = accept_scale_pct / 100
        self.work = 0
        self.in_solution = bytearray(n)
        self.tight = array("i", [0]) * n  # number of solution neighbors
        self.owner_sum = array("q", [0]) * n  # sum of their ids; the owner when tight == 1
        self.solution = []
        self.solution_pos = array("i", [-1]) * n
        self.free = list(range(n))  # outside the solution with tight == 0
        self.free_pos = array("i", range(n))
        self.stack = []  # solution vertices to re-examine for (1,2)-swaps
        self.queued = bytearray(n)
        self.age = array("q", [0]) * n  # iteration of the last membership change
        self.clock = 0
        self.log = None  # undo log during a trial step: v = insert, ~v = remove
        for v in initial:
            if not 0 <= v < n or self.free_pos[v] < 0:
                raise ValueError("Initial vertices must be distinct and independent.")
            self._insert(v)

    def run(self, iterations, on_improvement=None, work_budget=None):
        """Perturb and re-optimize until `iterations` steps or `work_budget` work; return results.

        Either limit may be None, but not both. The work check happens between
        iterations, so a run can exceed work_budget by part of one iteration.
        """
        if iterations is None and work_budget is None:
            raise ValueError("Give iterations, work_budget or both.")
        rng = self.rng
        self._local_search()
        current = len(self.solution)
        best = sorted(self.solution)
        if on_improvement is not None:
            on_improvement(0, current)
        accepted = rejected = 0
        iteration = 0
        while ((iterations is None or iteration < iterations)
               and (work_budget is None or self.work < work_budget)):
            iteration += 1
            self.clock = iteration
            self.log = []
            self._perturb()
            self._local_search()
            size = len(self.solution)
            if size >= current or rng.random() < min(
                    1.0, self.accept_scale / (1 + (current - size) * (len(best) - size))):
                current = size
                accepted += 1
                self.log = None
            else:
                self._undo()
                rejected += 1
            if current > len(best):
                best = sorted(self.solution)
                if on_improvement is not None:
                    on_improvement(iteration, current)
        return {
            "best": best,
            "best_size": len(best),
            "final_size": current,
            "iterations": iteration,
            "accepted": accepted,
            "rejected": rejected,
            "work": self.work,
        }

    def _local_search(self):
        stack = self.stack
        while True:
            if self.free:
                self._fill_free()
            if not stack:
                return
            x = stack.pop()
            self.queued[x] = 0
            if self.in_solution[x]:
                self._two_improve(x)

    def _fill_free(self):
        free = self.free
        while free:
            self._insert(free[self.rng.randbelow(len(free))])

    def _two_improve(self, x):
        """Replace x by two non-adjacent vertices whose only solution neighbor is x."""
        tight = self.tight
        neighbors = self.graph.neighbors(x)
        self.work += len(neighbors)
        candidates = [w for w in neighbors if tight[w] == 1]
        m = len(candidates)
        if m < 2:
            return False
        adjacent = self.graph.adjacent
        start = self.rng.randbelow(m)
        for a in range(m - 1):
            u = candidates[(start + a) % m]
            for b in range(a + 1, m):
                w = candidates[(start + b) % m]
                if not adjacent(u, w):
                    self._remove(x)
                    self._insert(u)
                    self._insert(w)
                    return True
        return False

    def _perturb(self):
        """Force k outside vertices in (k geometric, at most max_perturbation)."""
        if len(self.solution) == len(self.in_solution):
            return
        rng = self.rng
        graph = self.graph
        k = 1
        while k < self.max_perturbation and rng.random() < self.continue_probability:
            k += 1
        first = self._pick_outside()
        forced = [first]
        for _ in range(8 * k):
            if len(forced) == k:
                break
            v = first
            for _ in range(self.perturb_radius):  # a random walk, so within that distance of first
                hop = graph.neighbors(v)
                v = hop[rng.randbelow(len(hop))]
            if self.in_solution[v] or v in forced or any(graph.adjacent(v, u) for u in forced):
                continue
            forced.append(v)
        for v in forced:
            neighbors = graph.neighbors(v)
            self.work += len(neighbors)
            for w in neighbors:
                if self.in_solution[w]:
                    self._remove(w)
            self._insert(v)

    def _pick_outside(self):
        """Oldest of `tournament` random vertices outside the solution."""
        n = len(self.in_solution)
        chosen = -1
        for _ in range(self.tournament):
            v = self.rng.randbelow(n)
            while self.in_solution[v]:
                v = self.rng.randbelow(n)
            if chosen < 0 or self.age[v] < self.age[chosen]:
                chosen = v
        return chosen

    def _undo(self):
        log = self.log
        self.log = None
        for op in reversed(log):
            if op >= 0:
                self._remove(op)
            else:
                self._insert(~op)
        for x in self.stack:
            self.queued[x] = 0
        del self.stack[:]

    def _insert(self, v):
        self._unfree(v)
        self.in_solution[v] = 1
        self.solution_pos[v] = len(self.solution)
        self.solution.append(v)
        tight = self.tight
        owner_sum = self.owner_sum
        neighbors = self.graph.neighbors(v)
        self.work += len(neighbors)
        for w in neighbors:
            if tight[w] == 0:
                self._unfree(w)
            tight[w] += 1
            owner_sum[w] += v
        self.age[v] = self.clock
        if self.log is not None:
            self.log.append(v)
        self._push(v)

    def _remove(self, v):
        solution = self.solution
        position = self.solution_pos[v]
        last = solution.pop()
        if last != v:
            solution[position] = last
            self.solution_pos[last] = position
        self.solution_pos[v] = -1
        self.in_solution[v] = 0
        tight = self.tight
        owner_sum = self.owner_sum
        neighbors = self.graph.neighbors(v)
        self.work += len(neighbors)
        for w in neighbors:
            t = tight[w] - 1
            tight[w] = t
            owner_sum[w] -= v
            if t == 0:
                self._free(w)
            elif t == 1:
                self._push(owner_sum[w])
        self._free(v)
        self.age[v] = self.clock
        if self.log is not None:
            self.log.append(~v)

    def _push(self, x):
        if not self.queued[x]:
            self.queued[x] = 1
            self.stack.append(x)

    def _free(self, w):
        self.free_pos[w] = len(self.free)
        self.free.append(w)

    def _unfree(self, w):
        free = self.free
        position = self.free_pos[w]
        last = free.pop()
        if last != w:
            free[position] = last
            self.free_pos[last] = position
        self.free_pos[w] = -1

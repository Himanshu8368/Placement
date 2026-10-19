#pragma once
// functional_graph.hpp -- decomposition of a functional graph (every vertex has exactly
// one outgoing arc v -> f[v]) into cycles with trees hanging into them.
//   build(f):
//     cycles[c]        = the vertices of cycle c in arc order (cycles[c][i] -> cycles[c][i+1]);
//     cycle_id[v]      = the cycle that v eventually reaches (for ALL v);
//     dist_to_cycle[v] = number of steps until v first stands on a cycle (0 on a cycle);
//     entry[v]         = the first cycle vertex reached from v (v itself if on a cycle);
//     pos[v]           = index of v inside cycles[cycle_id[v]] (-1 if v is not on a cycle).
//   Then f^k(v) for k >= dist_to_cycle[v] is cycles[c][(pos[entry[v]] + k - d) % len].
// Complexity: O(n) time and memory, iterative (no recursion on long tails).
// Indexing  : vertices 0..n-1, f[v] in 0..n-1 (convert 1-indexed input first).
// Pitfalls  : every component contains exactly ONE cycle; a fixed point f[v] = v is a
//             cycle of length 1; do not use "visited" alone -- the current walk must be
//             told apart from walks of earlier starts (state 1 vs 2).
#include <bits/stdc++.h>
using namespace std;

struct FunctionalGraph {
    vector<vector<int>> cycles;
    vector<int> cycle_id, dist_to_cycle, entry, pos;

    void build(const vector<int>& f) {
        int n = f.size();
        cycles.clear();
        cycle_id.assign(n, -1);
        dist_to_cycle.assign(n, -1);
        entry.assign(n, -1);
        pos.assign(n, -1);
        vector<int> state(n, 0), path;       // 0 new, 1 on the current walk, 2 finished
        for (int s = 0; s < n; s++) {
            if (state[s]) continue;
            path.clear();
            int v = s;
            while (state[v] == 0) { state[v] = 1; path.push_back(v); v = f[v]; }
            if (state[v] == 1) {             // the walk bit its own tail: new cycle from v
                vector<int> cyc;
                int id = cycles.size();
                for (int x = v;;) {
                    pos[x] = cyc.size();
                    cyc.push_back(x);
                    cycle_id[x] = id; dist_to_cycle[x] = 0; entry[x] = x; state[x] = 2;
                    x = f[x];
                    if (x == v) break;
                }
                cycles.push_back(cyc);
            }
            for (int i = (int)path.size() - 1; i >= 0; i--) {   // tail vertices, backwards
                int x = path[i];
                if (state[x] == 2) continue; // a cycle vertex, done above
                cycle_id[x] = cycle_id[f[x]];
                dist_to_cycle[x] = dist_to_cycle[f[x]] + 1;
                entry[x] = entry[f[x]];
                state[x] = 2;
            }
        }
    }
};

#pragma once
// bfs.hpp -- breadth-first search from one source on an unweighted graph.
//   run(adj, s)  : dist[v] = fewest edges from s to v (-1 if unreachable),
//                  par[v]  = predecessor on one shortest path (-1 for s / unreachable),
//                  order   = vertices in the order they were popped.
//   path(t)      : the vertices s, ..., t of a shortest path; empty if t is unreachable.
// Complexity: O(n + m) time and O(n) memory.
// Indexing  : vertices 0..n-1.
// Pitfalls  : mark (set dist) when a vertex is PUSHED, never when it is popped -- see
//             Appendix "Testing Graph Code" for what goes wrong otherwise.
#include <bits/stdc++.h>
using namespace std;

struct BFS {
    vector<int> dist, par, order;

    void run(const vector<vector<int>>& adj, int s) {
        int n = adj.size();
        dist.assign(n, -1);
        par.assign(n, -1);
        order.clear();
        vector<int> q;               // a vector used as a queue: q[head..] is pending
        q.reserve(n);
        dist[s] = 0;
        q.push_back(s);
        for (size_t head = 0; head < q.size(); head++) {
            int u = q[head];
            order.push_back(u);
            for (int v : adj[u])
                if (dist[v] == -1) {             // first discovery = shortest
                    dist[v] = dist[u] + 1;
                    par[v] = u;
                    q.push_back(v);
                }
        }
    }

    vector<int> path(int t) const {
        if (dist[t] == -1) return {};
        vector<int> p;
        for (int v = t; v != -1; v = par[v]) p.push_back(v);
        reverse(p.begin(), p.end());
        return p;
    }
};

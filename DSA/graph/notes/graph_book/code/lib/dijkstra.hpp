#pragma once
// dijkstra.hpp -- single-source shortest paths, NON-NEGATIVE weights, lazy binary heap.
//   run(adj, s): adj[u] = {(v, w)}, w >= 0 (long long);
//                dist[v] = shortest distance (INF if unreachable), par[v] = predecessor.
//   path(t)    : vertices s, ..., t of a shortest path; empty if unreachable.
// Lazy      : instead of decrease-key, push a new (dist, v) pair and skip pairs whose
//             distance is stale when popped. The heap holds at most m + 1 entries.
// Complexity: O((n + m) log m).
// Indexing  : vertices 0..n-1.
// Pitfalls  : weights must be >= 0 (use Bellman-Ford otherwise); distances in long long;
//             the stale check "d != dist[u]" is what keeps the running time -- without
//             it a vertex can be expanded many times; INF = LLONG_MAX / 4 so that
//             INF + w does not overflow.
#include <bits/stdc++.h>
using namespace std;

struct Dijkstra {
    static constexpr long long INF = LLONG_MAX / 4;
    vector<long long> dist;
    vector<int> par;

    void run(const vector<vector<pair<int, long long>>>& adj, int s) {
        int n = adj.size();
        dist.assign(n, INF);
        par.assign(n, -1);
        priority_queue<pair<long long, int>, vector<pair<long long, int>>, greater<pair<long long, int>>> pq;
        dist[s] = 0;
        pq.push({0, s});
        while (!pq.empty()) {
            long long d = pq.top().first;
            int u = pq.top().second;
            pq.pop();
            if (d != dist[u]) continue;          // stale entry
            for (const auto& e : adj[u]) {
                int v = e.first;
                long long nd = d + e.second;
                if (nd < dist[v]) {
                    dist[v] = nd;
                    par[v] = u;
                    pq.push({nd, v});
                }
            }
        }
    }

    vector<int> path(int t) const {
        if (dist[t] >= INF) return {};
        vector<int> p;
        for (int v = t; v != -1; v = par[v]) p.push_back(v);
        reverse(p.begin(), p.end());
        return p;
    }
};
constexpr long long Dijkstra::INF;

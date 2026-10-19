#pragma once
// bellman_ford.hpp -- shortest paths with negative weights; negative-cycle detection.
//   BellmanFord::run(n, edges, s): edges are directed arcs {u, v, w}. Afterwards
//       dist[v] = INF   if v is unreachable from s,
//               = -INF  if v is reachable through a negative cycle (no shortest path),
//               = the shortest distance otherwise;   par[v] = predecessor (finite dist).
//       Returns true iff some negative cycle is reachable from s.
//   find_negative_cycle(n, edges): ANY negative cycle in the whole graph (as if a virtual
//       source reached every vertex), as edge indices e0, e1, ..., e(k-1) in cycle order
//       (head of e_i = tail of e_(i+1), head of the last = tail of e0); empty if none.
// Complexity: O(n * m).
// Indexing  : vertices 0..n-1, edge indices 0..m-1. For an undirected graph add both
//             arcs -- but then ONE negative undirected edge is already a negative cycle.
// Pitfalls  : "relaxation in round n" proves a negative cycle exists, but the vertex
//             relaxed need not lie on it: walk n steps back along par first; propagate
//             -INF to everything reachable from a cycle, not only to the cycle itself.
#include <bits/stdc++.h>
using namespace std;

#ifndef GRAPHLIB_EDGE
#define GRAPHLIB_EDGE
struct Edge { int u, v; long long w; };
#endif

struct BellmanFord {
    static constexpr long long INF = LLONG_MAX / 4;
    vector<long long> dist;
    vector<int> par;

    bool run(int n, const vector<Edge>& edges, int s) {
        dist.assign(n, INF);
        par.assign(n, -1);
        dist[s] = 0;
        for (int round = 0; round < n - 1; round++) {
            bool changed = false;
            for (const auto& e : edges)
                if (dist[e.u] < INF && dist[e.u] + e.w < dist[e.v]) {
                    dist[e.v] = dist[e.u] + e.w;
                    par[e.v] = e.u;
                    changed = true;
                }
            if (!changed) break;
        }
        bool neg = false;
        for (int round = 0; round < n; round++)      // n more rounds spread -INF
            for (const auto& e : edges)
                if (dist[e.u] < INF && dist[e.v] > -INF &&
                    (dist[e.u] == -INF || dist[e.u] + e.w < dist[e.v])) {
                    dist[e.v] = -INF;
                    neg = true;
                }
        return neg;
    }
};
constexpr long long BellmanFord::INF;

inline vector<int> find_negative_cycle(int n, const vector<Edge>& edges) {
    vector<long long> dist(n, 0);                    // virtual source: everything at 0
    vector<int> pe(n, -1);                           // edge used to reach v last
    int x = -1;
    for (int round = 0; round < n; round++) {
        x = -1;
        for (int i = 0; i < (int)edges.size(); i++) {
            const Edge& e = edges[i];
            if (dist[e.u] + e.w < dist[e.v]) {
                dist[e.v] = dist[e.u] + e.w;
                pe[e.v] = i;
                x = e.v;
            }
        }
        if (x == -1) return {};                      // converged: no negative cycle
    }
    for (int i = 0; i < n; i++) x = edges[pe[x]].u;  // now x is surely ON the cycle
    vector<int> cyc;
    for (int v = x;;) {
        cyc.push_back(pe[v]);
        v = edges[pe[v]].u;
        if (v == x) break;
    }
    reverse(cyc.begin(), cyc.end());
    return cyc;
}

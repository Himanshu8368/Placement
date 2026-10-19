#pragma once
// zero_one_bfs.hpp -- shortest paths when every edge weight is 0 or 1 (a deque, not a heap).
//   run(adj, s): adj[u] = {(v, w)} with w in {0, 1};
//                dist[v] = shortest distance (INF if unreachable), par[v] = predecessor.
// Idea      : a 0-edge keeps the distance, so its endpoint goes to the FRONT of the deque;
//             a 1-edge goes to the BACK. The deque always holds at most two distinct
//             distances d and d+1, in sorted order -- a Dijkstra without the log.
// Complexity: O(n + m).
// Indexing  : vertices 0..n-1.
// Pitfalls  : a vertex can be pushed twice (first by a 1-edge, then improved by a 0-edge);
//             skip stale entries with the "done" check, and relax with a strict "<".
#include <bits/stdc++.h>
using namespace std;

struct ZeroOneBFS {
    static constexpr int INF = INT_MAX;
    vector<int> dist, par;

    void run(const vector<vector<pair<int, int>>>& adj, int s) {
        int n = adj.size();
        dist.assign(n, INF);
        par.assign(n, -1);
        vector<char> done(n, 0);
        deque<int> dq;
        dist[s] = 0;
        dq.push_back(s);
        while (!dq.empty()) {
            int u = dq.front();
            dq.pop_front();
            if (done[u]) continue;           // stale copy
            done[u] = 1;
            for (const auto& e : adj[u]) {
                int v = e.first, w = e.second;
                if (dist[u] + w < dist[v]) {
                    dist[v] = dist[u] + w;
                    par[v] = u;
                    if (w == 0) dq.push_front(v); else dq.push_back(v);
                }
            }
        }
    }
};
constexpr int ZeroOneBFS::INF;

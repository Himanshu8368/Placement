#pragma once
// multi_bfs.hpp -- multi-source BFS: distance to the NEAREST of several sources.
//   run(adj, sources): dist[v] = min over sources s of dist(s, v)  (-1 if none reaches v),
//                      src[v]  = one nearest source (the one whose wave arrived first),
//                      par[v]  = predecessor on a shortest path from src[v].
// Idea      : put all sources in the queue at distance 0 -- equivalent to one BFS from a
//             virtual super-source joined to every source by an edge of length 0.
// Complexity: O(n + m), independent of the number of sources.
// Indexing  : vertices 0..n-1; duplicate sources are ignored.
// Pitfalls  : running one BFS per source is O(k (n + m)) -- usually the TLE version.
#include <bits/stdc++.h>
using namespace std;

struct MultiBFS {
    vector<int> dist, src, par;

    void run(const vector<vector<int>>& adj, const vector<int>& sources) {
        int n = adj.size();
        dist.assign(n, -1);
        src.assign(n, -1);
        par.assign(n, -1);
        vector<int> q;
        for (int s : sources)
            if (dist[s] == -1) {
                dist[s] = 0;
                src[s] = s;
                q.push_back(s);
            }
        for (size_t head = 0; head < q.size(); head++) {
            int u = q[head];
            for (int v : adj[u])
                if (dist[v] == -1) {
                    dist[v] = dist[u] + 1;
                    src[v] = src[u];
                    par[v] = u;
                    q.push_back(v);
                }
        }
    }
};

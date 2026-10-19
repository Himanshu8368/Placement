#pragma once
// floyd_warshall.hpp -- all-pairs shortest paths on a distance matrix.
//   make_matrix(n)         : n x n matrix, 0 on the diagonal, FW_INF elsewhere;
//   floyd_warshall(d)      : in place, d[i][j] = shortest distance (FW_INF = no path);
//                            negative weights allowed;
//   has_negative_cycle(d)  : after floyd_warshall, true iff some d[i][i] < 0
//                            (vertex i lies on / reaches back through a negative cycle).
// Idea      : after round k, d[i][j] is the shortest path using intermediate vertices
//             0..k only. k MUST be the outermost loop.
// Complexity: O(n^3) time, O(n^2) memory -- n <= ~500 in practice.
// Indexing  : vertices 0..n-1.
// Pitfalls  : take the MIN when adding parallel edges (d[u][v] = min(d[u][v], w));
//             never add to FW_INF (the guard below); with negative cycles the values can
//             fall exponentially, so they are clamped at -FW_INF.
#include <bits/stdc++.h>
using namespace std;

const long long FW_INF = LLONG_MAX / 4;

inline vector<vector<long long>> make_matrix(int n) {
    vector<vector<long long>> d(n, vector<long long>(n, FW_INF));
    for (int i = 0; i < n; i++) d[i][i] = 0;
    return d;
}

inline void floyd_warshall(vector<vector<long long>>& d) {
    int n = d.size();
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++) {
            if (d[i][k] == FW_INF) continue;
            for (int j = 0; j < n; j++) {
                if (d[k][j] == FW_INF) continue;
                long long nd = max(d[i][k] + d[k][j], -FW_INF);
                if (nd < d[i][j]) d[i][j] = nd;
            }
        }
}

inline bool has_negative_cycle(const vector<vector<long long>>& d) {
    for (int i = 0; i < (int)d.size(); i++)
        if (d[i][i] < 0) return true;
    return false;
}

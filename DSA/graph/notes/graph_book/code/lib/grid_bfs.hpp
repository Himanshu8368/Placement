#pragma once
// grid_bfs.hpp -- BFS on a character grid without building the graph.
//   run(g, starts, eight): g[r][c] == wall ('#' by default) is blocked; every start
//                          cell gets distance 0 (multi-source); moves are the 4
//                          side neighbours, or all 8 neighbours if eight == true.
//   dist[r][c] = fewest moves (-1 if unreachable or a wall);
//   par[r][c]  = the cell (r*C + c) we came from, -1 for starts / unreached;
//   path(r, c) = list of (row, col) cells from a start to (r, c); empty if unreachable.
// Complexity: O(R * C * k), k = 4 or 8.
// Indexing  : rows 0..R-1, columns 0..C-1; a cell id is r*C + c.
// Pitfalls  : check bounds BEFORE reading g[nr][nc]; with 8 moves a diagonal step may
//             squeeze between two walls -- forbid it explicitly if the statement says so.
#include <bits/stdc++.h>
using namespace std;

struct GridBFS {
    int R = 0, C = 0;
    vector<vector<int>> dist, par;

    void run(const vector<string>& g, const vector<pair<int, int>>& starts,
             bool eight = false, char wall = '#') {
        static const int dr[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
        static const int dc[8] = {0, 0, -1, 1, -1, 1, -1, 1};
        R = g.size();
        C = R ? g[0].size() : 0;
        dist.assign(R, vector<int>(C, -1));
        par.assign(R, vector<int>(C, -1));
        int k = eight ? 8 : 4;
        vector<pair<int, int>> q;
        for (const auto& s : starts)
            if (g[s.first][s.second] != wall && dist[s.first][s.second] == -1) {
                dist[s.first][s.second] = 0;
                q.push_back(s);
            }
        for (size_t head = 0; head < q.size(); head++) {
            int r = q[head].first, c = q[head].second;
            for (int d = 0; d < k; d++) {
                int nr = r + dr[d], nc = c + dc[d];
                if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;   // bounds first
                if (g[nr][nc] == wall || dist[nr][nc] != -1) continue;
                dist[nr][nc] = dist[r][c] + 1;
                par[nr][nc] = r * C + c;
                q.push_back({nr, nc});
            }
        }
    }

    vector<pair<int, int>> path(int r, int c) const {
        if (dist[r][c] == -1) return {};
        vector<pair<int, int>> p;
        for (int id = r * C + c; id != -1; id = par[id / C][id % C]) p.push_back({id / C, id % C});
        reverse(p.begin(), p.end());
        return p;
    }
};

#pragma once
// dfs_recursive.hpp -- the textbook recursive DFS, same output as IterDFS.
//   run(adj) / run(adj, root), order[], parent[], tin[], tout[] exactly as in
//   dfs_iterative.hpp (one clock for entry and exit events).
// Complexity: O(n + m) time; recursion depth = longest DFS-tree path, up to n - 1.
// Indexing  : vertices 0..n-1.
// DEPTH NOTE: every open call costs a stack frame (~50-100 bytes here). A path of
//             10^6 vertices needs ~10^6 frames: fine on Codeforces (256 MB stack),
//             a crash (stack overflow, "runtime error") on a default 1-8 MB stack
//             (Windows, many judges, LeetCode). If depth can exceed ~10^5, use IterDFS.
#include <bits/stdc++.h>
using namespace std;

struct RecDFS {
    vector<int> order, parent, tin, tout;
    const vector<vector<int>>* g = nullptr;
    int timer = 0;

    void dfs(int u) {
        tin[u] = timer++;
        order.push_back(u);
        for (int v : (*g)[u])
            if (tin[v] == -1) {
                parent[v] = u;
                dfs(v);
            }
        tout[u] = timer++;
    }

    void run(const vector<vector<int>>& adj, int root = -1) {
        int n = adj.size();
        g = &adj;
        timer = 0;
        order.clear();
        parent.assign(n, -1);
        tin.assign(n, -1);
        tout.assign(n, -1);
        if (root >= 0) dfs(root);
        else
            for (int s = 0; s < n; s++)
                if (tin[s] == -1) dfs(s);
    }
};

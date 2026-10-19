#include <bits/stdc++.h>
using namespace std;

// Pitfall 3: parallel edges. Bridge finding (low-link) that skips the parent VERTEX
// (buggy) versus the parent EDGE (correct).
// Input: n m, then m undirected edges (1-indexed). Prints the bridges as edge numbers.
int n, m, timer_;
vector<vector<pair<int, int>>> adj;   // (neighbour, edge id)
vector<int> tin, low;
vector<int> bridges;

void dfs(int u, int parent, int pedge, bool skip_edge) {
    tin[u] = low[u] = timer_++;
    for (auto& e : adj[u]) {
        int v = e.first, id = e.second;
        if (skip_edge ? id == pedge : v == parent) continue;
        if (tin[v] != -1) low[u] = min(low[u], tin[v]);
        else {
            dfs(v, u, id, skip_edge);
            low[u] = min(low[u], low[v]);
            if (low[v] > tin[u]) bridges.push_back(id + 1);
        }
    }
}

int main() {
    cin >> n >> m;
    adj.assign(n + 1, {});
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back({v, i});
        if (u != v) adj[v].push_back({u, i});
    }
    for (int version = 0; version < 2; version++) {
        tin.assign(n + 1, -1); low.assign(n + 1, -1); bridges.clear(); timer_ = 0;
        for (int s = 1; s <= n; s++) if (tin[s] == -1) dfs(s, 0, -1, version == 1);
        sort(bridges.begin(), bridges.end());
        cout << (version ? "skip parent edge  : bridges" : "skip parent vertex: bridges");
        for (int b : bridges) cout << " " << b;
        cout << "\n";
    }
}

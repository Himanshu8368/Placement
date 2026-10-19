#include <bits/stdc++.h>
using namespace std;

// Pitfall 2: disconnected input. Bipartite check that only starts a BFS from vertex 1.
// Input: n m, then m undirected edges (1-indexed). Prints BUGGY and CORRECT answers.
int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }
    auto check = [&](bool all_starts) {
        vector<int> col(n + 1, -1);
        for (int s = 1; s <= n; s++) {
            if (col[s] != -1 || (!all_starts && s != 1)) continue;
            col[s] = 0;
            queue<int> q;
            q.push(s);
            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (int v : adj[u]) {
                    if (col[v] == -1) { col[v] = col[u] ^ 1; q.push(v); }
                    else if (col[v] == col[u]) return false;
                }
            }
        }
        return true;
    };
    cout << "buggy (start at 1 only): " << (check(false) ? "YES" : "NO") << "\n";
    cout << "correct (every start)  : " << (check(true) ? "YES" : "NO") << "\n";
}

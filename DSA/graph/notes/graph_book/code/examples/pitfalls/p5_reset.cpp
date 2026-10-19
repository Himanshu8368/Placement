#include <bits/stdc++.h>
using namespace std;

// Pitfall 5: multi-test input with global arrays that are not reset.
// Input: t, then t tests "n m" + m undirected edges. Prints the number of components.
const int MAXN = 100;
vector<int> adj[MAXN + 1];
bool seen[MAXN + 1];

void dfs(int u) {
    seen[u] = true;
    for (int v : adj[u]) if (!seen[v]) dfs(v);
}

int main() {
    int t;
    cin >> t;
    while (t--) {
        int n, m;
        cin >> n >> m;
        // BUG: forgot  for (v = 1..n) { adj[v].clear(); seen[v] = false; }
        for (int i = 0; i < m; i++) {
            int u, v;
            cin >> u >> v;
            adj[u].push_back(v);
            adj[v].push_back(u);
        }
        int comps = 0;
        for (int v = 1; v <= n; v++)
            if (!seen[v]) { comps++; dfs(v); }
        cout << comps << "\n";
    }
}

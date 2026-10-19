#include <bits/stdc++.h>
using namespace std;

// BUGGY BFS: vertices are marked visited when POPPED, not when pushed.
// Input : n m, then m undirected edges "u v" (1-indexed). Output: dist(1, v) for v = 1..n
//         (-1 if unreachable).
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
    vector<int> dist(n + 1, -1);
    vector<bool> visited(n + 1, false);
    queue<int> q;
    dist[1] = 0;
    q.push(1);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        visited[u] = true;                 // BUG: marked too late
        for (int v : adj[u])
            if (!visited[v]) {             // v may already be waiting in the queue ...
                dist[v] = dist[u] + 1;     // ... and its distance is overwritten
                q.push(v);
            }
    }
    for (int v = 1; v <= n; v++) cout << dist[v] << (v == n ? '\n' : ' ');
}

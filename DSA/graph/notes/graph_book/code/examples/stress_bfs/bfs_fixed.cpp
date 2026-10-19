#include <bits/stdc++.h>
using namespace std;

// FIXED BFS: a vertex is marked the moment it is PUSHED (dist[v] != -1 means seen).
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
    queue<int> q;
    dist[1] = 0;
    q.push(1);
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int v : adj[u])
            if (dist[v] == -1) {           // first discovery: mark now, never again
                dist[v] = dist[u] + 1;
                q.push(v);
            }
    }
    for (int v = 1; v <= n; v++) cout << dist[v] << (v == n ? '\n' : ' ');
}

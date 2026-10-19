#include <bits/stdc++.h>
using namespace std;

// Pitfall 4: 1-indexed input stored in a 0-indexed array of size n.
// Input: n m, then m undirected edges (1-indexed). Prints the degree of every vertex.
int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> adj(n);          // BUG: vertices are 1..n, so adj needs n + 1 slots
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);             // adj[n] is out of bounds
        adj[v].push_back(u);
    }
    for (int v = 1; v <= n; v++) cout << "deg(" << v << ") = " << adj[v].size() << "\n";
}

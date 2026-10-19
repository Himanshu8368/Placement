#include <bits/stdc++.h>
using namespace std;

// Pitfall 6: distances in int. Dijkstra with int and with long long on the same input.
// Input: n m, then m undirected edges "u v w" (1-indexed). Prints dist(1, n).
template <class T>
T dijkstra(int n, const vector<vector<pair<int, int>>>& adj, T INF) {
    vector<T> d(n + 1, INF);
    priority_queue<pair<T, int>, vector<pair<T, int>>, greater<pair<T, int>>> pq;
    d[1] = 0;
    pq.push({0, 1});
    while (!pq.empty()) {
        T du = pq.top().first;
        int u = pq.top().second;
        pq.pop();
        if (du != d[u]) continue;
        for (auto& e : adj[u])
            if (du + e.second < d[e.first]) {   // int: du + w can overflow
                d[e.first] = du + e.second;
                pq.push({d[e.first], e.first});
            }
    }
    return d[n];
}

int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<pair<int, int>>> adj(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v, w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
    }
    cout << "int       : " << dijkstra<int>(n, adj, INT_MAX) << "\n";
    cout << "long long : " << dijkstra<long long>(n, adj, LLONG_MAX / 4) << "\n";
}

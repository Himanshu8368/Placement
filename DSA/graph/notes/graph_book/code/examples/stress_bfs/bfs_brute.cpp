#include <bits/stdc++.h>
using namespace std;

// Brute force for the same task: relax every edge n times (Bellman-Ford with weight 1).
// Too slow for big inputs (O(n m)), obviously correct.
int main() {
    int n, m;
    cin >> n >> m;
    vector<pair<int, int>> e(m);
    for (auto& x : e) cin >> x.first >> x.second;
    const int INF = INT_MAX / 2;
    vector<int> d(n + 1, INF);
    d[1] = 0;
    for (int round = 0; round < n; round++)
        for (auto& x : e) {
            d[x.second] = min(d[x.second], d[x.first] + 1);
            d[x.first] = min(d[x.first], d[x.second] + 1);
        }
    for (int v = 1; v <= n; v++) cout << (d[v] >= INF ? -1 : d[v]) << (v == n ? '\n' : ' ');
}

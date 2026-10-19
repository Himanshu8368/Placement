#include <bits/stdc++.h>
using namespace std;

// Pitfall 8: an undirected edge stored in one direction only.
// Input: n m, then m undirected edges (1-indexed). Prints the vertices reachable from 1.
int main() {
    int n, m;
    cin >> n >> m;
    vector<vector<int>> one(n + 1), both(n + 1);
    for (int i = 0; i < m; i++) {
        int u, v;
        cin >> u >> v;
        one[u].push_back(v);                         // BUG: the edge is undirected
        both[u].push_back(v);
        both[v].push_back(u);
    }
    for (auto* g : {&one, &both}) {
        vector<char> seen(n + 1, 0);
        vector<int> st = {1};
        seen[1] = 1;
        while (!st.empty()) {
            int u = st.back(); st.pop_back();
            for (int v : (*g)[u]) if (!seen[v]) { seen[v] = 1; st.push_back(v); }
        }
        cout << (g == &one ? "one direction :" : "both         :");
        for (int v = 1; v <= n; v++) if (seen[v]) cout << " " << v;
        cout << "\n";
    }
}

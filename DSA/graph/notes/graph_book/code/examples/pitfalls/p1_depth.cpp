#include <bits/stdc++.h>
using namespace std;

// Pitfall 1: recursion depth. Recursive DFS on a path 0 - 1 - ... - (n-1).
// Run: p1_depth N   (default stack of the executable)
vector<vector<int>> adj;
vector<char> seen;
int deepest = 0;

void dfs(int u, int depth) {
    seen[u] = 1;
    deepest = max(deepest, depth);
    for (int v : adj[u])
        if (!seen[v]) dfs(v, depth + 1);
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1000000;
    adj.assign(n, vector<int>());
    for (int i = 0; i + 1 < n; i++) { adj[i].push_back(i + 1); adj[i + 1].push_back(i); }
    seen.assign(n, 0);
    dfs(0, 0);
    cout << "n = " << n << ": recursion depth " << deepest << ", finished\n";
}

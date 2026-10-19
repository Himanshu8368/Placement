#include <bits/stdc++.h>
using namespace std;

// Pitfall 7: marking visited when popped instead of when pushed -- the cost.
// On the complete graph K_n, count queue pushes of both versions (the distances are
// set only once here, so only the running time differs).
int main() {
    for (int n : {10, 100, 1000, 3000}) {
        long long push_pop = 0, push_push = 0;
        {   // mark when popped
            vector<char> vis(n, 0);
            queue<int> q;
            q.push(0); push_pop++;
            while (!q.empty()) {
                int u = q.front(); q.pop();
                if (vis[u]) continue;
                vis[u] = 1;
                for (int v = 0; v < n; v++)
                    if (v != u && !vis[v]) { q.push(v); push_pop++; }
            }
        }
        {   // mark when pushed
            vector<char> vis(n, 0);
            queue<int> q;
            vis[0] = 1; q.push(0); push_push++;
            while (!q.empty()) {
                int u = q.front(); q.pop();
                for (int v = 0; v < n; v++)
                    if (v != u && !vis[v]) { vis[v] = 1; q.push(v); push_push++; }
            }
        }
        cout << "K_" << n << ": pushes when marking on pop = " << push_pop
             << ", on push = " << push_push << "\n";
    }
}

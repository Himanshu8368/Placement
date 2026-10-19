#pragma once
// check.hpp -- shared helpers for the library tests: failure reporting and brute-force
// references that are too slow for contests but obviously correct on tiny graphs.
// Each test defines TEST_NAME before including this file.
#include "../gen/gen.hpp"

static string g_case;      // description of the current input, printed on failure

#define CHECK(cond, msg)                                                               \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            cout << "FAIL " << TEST_NAME << "\n  input: " << g_case << "\n  " << msg   \
                 << "  [" #cond "]  (line " << __LINE__ << ")\n";                      \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

inline void report_ok(int cases) { cout << "OK " << TEST_NAME << " " << cases << "\n"; }

// reach[u][v] = 1 iff there is a walk from u to v with >= 0 arcs (Floyd-Warshall closure)
inline vector<vector<int>> closure(int n, const vector<pair<int, int>>& e, bool directed) {
    vector<vector<int>> r(n, vector<int>(n, 0));
    for (int i = 0; i < n; i++) r[i][i] = 1;
    for (auto& x : e) { r[x.first][x.second] = 1; if (!directed) r[x.second][x.first] = 1; }
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            if (r[i][k])
                for (int j = 0; j < n; j++)
                    if (r[k][j]) r[i][j] = 1;
    return r;
}

// strict[u][v] = 1 iff there is a walk from u to v with >= 1 arc
inline vector<vector<int>> strict_closure(int n, const vector<pair<int, int>>& e) {
    vector<vector<int>> r(n, vector<int>(n, 0));
    for (auto& x : e) r[x.first][x.second] = 1;
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            if (r[i][k])
                for (int j = 0; j < n; j++)
                    if (r[k][j]) r[i][j] = 1;
    return r;
}

// unweighted all-pairs distances (-1 = unreachable), Floyd-Warshall
inline vector<vector<int>> apsp_unweighted(int n, const vector<pair<int, int>>& e, bool directed) {
    const int INF = 1 << 29;
    vector<vector<int>> d(n, vector<int>(n, INF));
    for (int i = 0; i < n; i++) d[i][i] = 0;
    for (auto& x : e) {
        if (x.first != x.second) d[x.first][x.second] = 1;
        if (!directed && x.first != x.second) d[x.second][x.first] = 1;
    }
    for (int k = 0; k < n; k++)
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++) d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
    for (auto& row : d)
        for (auto& x : row)
            if (x >= INF) x = -1;
    return d;
}

inline bool has_arc(const vector<vector<int>>& adj, int u, int v) {
    return find(adj[u].begin(), adj[u].end(), v) != adj[u].end();
}

// naive component labels (label = smallest vertex of the component)
inline vector<int> naive_labels(int n, const vector<pair<int, int>>& e) {
    auto r = closure(n, e, false);
    vector<int> lab(n);
    for (int v = 0; v < n; v++)
        for (int u = 0; u < n; u++)
            if (r[u][v]) { lab[v] = u; break; }
    return lab;
}

// Exhaustive search over SIMPLE paths / cycles of a tiny weighted digraph (n <= 7).
const long long BR_INF = LLONG_MAX / 4;
struct SimplePaths {
    int n;
    vector<vector<pair<int, long long>>> adj;
    vector<vector<long long>> best;     // best[s][t] over simple paths, BR_INF if none
    vector<char> on_neg_cycle;          // vertex lies on some negative simple cycle
    bool any_neg = false;
    vector<char> used;
    int start = 0;
    vector<int> stk;

    void go(int u, long long len) {
        best[start][u] = min(best[start][u], len);
        for (auto& e : adj[u]) {
            int v = e.first;
            if (v == start && len + e.second < 0) {
                any_neg = true;                          // closes a negative cycle
                for (int x : stk) on_neg_cycle[x] = 1;
            }
            if (used[v]) continue;
            used[v] = 1; stk.push_back(v);
            go(v, len + e.second);
            used[v] = 0; stk.pop_back();
        }
    }
    SimplePaths(int n_, const vector<Edge>& e, bool directed) : n(n_), adj(n_) {
        for (auto& x : e) {
            adj[x.u].push_back({x.v, x.w});
            if (!directed) adj[x.v].push_back({x.u, x.w});
        }
        best.assign(n, vector<long long>(n, BR_INF));
        on_neg_cycle.assign(n, 0);
        used.assign(n, 0);
        for (start = 0; start < n; start++) {
            used[start] = 1; stk.assign(1, start);
            go(start, 0);
            used[start] = 0;
        }
    }
};

// minimum spanning forest weight by trying every subset of edges (m <= ~12)
inline pair<long long, int> brute_msf(int n, const vector<Edge>& e) {
    int m = e.size(), comps = n;
    {   // number of components of the whole graph
        vector<pair<int, int>> pe;
        for (auto& x : e) pe.push_back({x.u, x.v});
        auto lab = naive_labels(n, pe);
        comps = 0;
        for (int v = 0; v < n; v++) comps += lab[v] == v;
    }
    long long best = LLONG_MAX;
    for (int mask = 0; mask < (1 << m); mask++) {
        if (__builtin_popcount(mask) != n - comps) continue;
        vector<int> lab(n);
        iota(lab.begin(), lab.end(), 0);
        bool ok = true;
        long long w = 0;
        for (int i = 0; i < m && ok; i++)
            if (mask >> i & 1) {
                int a = lab[e[i].u], b = lab[e[i].v];
                if (a == b) { ok = false; break; }       // cycle (or self-loop)
                for (auto& x : lab) if (x == b) x = a;
                w += e[i].w;
            }
        if (ok) best = min(best, w);
    }
    return {best, comps};
}

// shared DFS checks for IterDFS / RecDFS
template <class D>
void check_dfs(const D& d, const vector<vector<int>>& adj, const vector<vector<int>>& reach, int root) {
    int n = adj.size();
    for (int v = 0; v < n; v++) {
        bool seen = d.tin[v] != -1;
        if (root >= 0) CHECK(seen == (bool)reach[root][v], "visited set differs from reachability, v=" << v);
        else CHECK(seen, "full DFS must visit every vertex, v=" << v);
        if (!seen) continue;
        CHECK(d.tin[v] < d.tout[v], "tin < tout, v=" << v);
        if (d.parent[v] != -1) {
            int p = d.parent[v];
            CHECK(has_arc(adj, p, v), "parent arc missing " << p << "->" << v);
            CHECK(d.tin[p] < d.tin[v] && d.tout[v] < d.tout[p], "child interval inside parent");
        } else if (root >= 0) CHECK(v == root, "only the root has no parent");
    }
    // order = vertices sorted by tin, and intervals are nested or disjoint
    for (size_t i = 0; i + 1 < d.order.size(); i++) CHECK(d.tin[d.order[i]] < d.tin[d.order[i + 1]], "order by tin");
    for (int u = 0; u < n; u++)
        for (int v = 0; v < n; v++)
            if (d.tin[u] != -1 && d.tin[v] != -1 && d.tin[u] < d.tin[v])
                CHECK(d.tout[v] < d.tout[u] || d.tout[u] < d.tin[v], "parenthesis property");
    // DFS property: an arc u->v never goes to a vertex discovered AFTER u finished
    for (int u = 0; u < n; u++)
        if (d.tin[u] != -1)
            for (int v : adj[u]) CHECK(d.tin[v] != -1 && d.tin[v] < d.tout[u], "white-path property, arc " << u << "->" << v);
}

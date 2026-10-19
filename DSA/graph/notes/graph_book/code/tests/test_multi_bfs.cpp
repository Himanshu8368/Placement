// test_multi_bfs.cpp -- multi-source BFS against "minimum over sources" of Floyd-Warshall
// distances; src[v] really is a nearest source; par[] is a shortest-path tree.
#define TEST_NAME "multi_bfs"
#include "check.hpp"
#include "../lib/multi_bfs.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 5000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 10), m = g.rnd(0, 16), k = g.rnd(1, 4);
        bool directed = g.rnd(0, 1);
        auto e = g.graph(n, m, true, true, directed);
        vector<int> src;
        for (int i = 0; i < k; i++) src.push_back(g.rnd(0, n - 1));   // duplicates possible
        g_case = dump_edges(n, e) + (directed ? " directed" : " undirected") + " sources:";
        for (int s : src) g_case += " " + to_string(s);
        auto adj = to_adj(n, e, directed);
        auto D = apsp_unweighted(n, e, directed);
        MultiBFS mb;
        mb.run(adj, src);
        for (int v = 0; v < n; v++) {
            int best = -1;
            for (int s : src)
                if (D[s][v] != -1 && (best == -1 || D[s][v] < best)) best = D[s][v];
            CHECK(mb.dist[v] == best, "dist[" << v << "] = " << mb.dist[v] << ", expected " << best);
            if (best == -1) continue;
            CHECK(find(src.begin(), src.end(), mb.src[v]) != src.end(), "src[v] is not a source");
            CHECK(D[mb.src[v]][v] == best, "src[v] is not a nearest source");
            if (mb.dist[v] > 0) {
                int p = mb.par[v];
                CHECK(has_arc(adj, p, v) && mb.dist[p] == mb.dist[v] - 1 && mb.src[p] == mb.src[v], "par[] tree");
            }
        }
    }
    report_ok(cases);
}

// test_graph.cpp -- Graph / WGraph / read_graph / CSR against a naive edge-list build.
#define TEST_NAME "graph"
#include "check.hpp"
#include "../lib/graph.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 3000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 8), m = g.rnd(0, 14);
        bool directed = g.rnd(0, 1);
        auto e = g.graph(n, m, true, true, directed);   // self-loops and multi-edges allowed
        g_case = dump_edges(n, e) + (directed ? " directed" : " undirected");
        auto ref = to_adj(n, e, directed);

        Graph G(n);
        for (auto& x : e) { if (directed) G.add_arc(x.first, x.second); else G.add_edge(x.first, x.second); }
        CHECK(G.adj == ref, "Graph adjacency lists differ");

        ostringstream os;                                 // 1-indexed text input
        for (auto& x : e) os << x.first + 1 << " " << x.second + 1 << "\n";
        istringstream is(os.str());
        Graph R = read_graph(is, n, e.size(), directed, 1);
        CHECK(R.adj == ref, "read_graph differs");

        CSR C(n, e, directed);
        CHECK((int)C.to.size() == (directed ? 1 : 2) * (int)e.size(), "CSR size");
        for (int u = 0; u < n; u++) {
            vector<int> nb(C.to.begin() + C.start[u], C.to.begin() + C.start[u + 1]);
            CHECK(nb == ref[u], "CSR neighbours of " << u << " differ (order must be input order)");
            CHECK(C.deg(u) == (int)ref[u].size(), "CSR degree");
        }

        auto we = g.weighted(e, -5, 5);
        auto wref = to_wadj(n, we, directed);
        WGraph W(n);
        ostringstream ws;
        for (auto& x : we) {
            if (directed) W.add_arc(x.u, x.v, x.w); else W.add_edge(x.u, x.v, x.w);
            ws << x.u + 1 << " " << x.v + 1 << " " << x.w << "\n";
        }
        CHECK(W.adj == wref, "WGraph differs");
        istringstream wis(ws.str());
        CHECK(read_wgraph(wis, n, we.size(), directed, 1).adj == wref, "read_wgraph differs");
    }
    report_ok(cases);
}

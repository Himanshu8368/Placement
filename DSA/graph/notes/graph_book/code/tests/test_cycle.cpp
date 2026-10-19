// test_cycle.cpp -- undirected: a cycle exists iff m > n - components (true for multigraphs
// with loops too); directed: iff some vertex reaches itself by >= 1 arc (closure). Every
// returned cycle is checked edge by edge.
#define TEST_NAME "cycle"
#include "check.hpp"
#include "../lib/cycle.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 4000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 9), m = g.rnd(0, 10);
        // undirected
        auto e = g.graph(n, m, g.rnd(0, 2) == 0, g.rnd(0, 1));
        g_case = dump_edges(n, e) + " undirected";
        auto lab = naive_labels(n, e);
        int comps = 0;
        for (int v = 0; v < n; v++) comps += lab[v] == v;
        bool brute = (int)e.size() > n - comps;
        auto c = find_cycle_undirected(n, e);
        CHECK(!c.verts.empty() == brute, "found=" << !c.verts.empty() << " expected " << brute);
        int k = c.verts.size();
        CHECK((int)c.ids.size() == k, "ids size");
        CHECK((int)set<int>(c.verts.begin(), c.verts.end()).size() == k, "repeated vertex");
        CHECK((int)set<int>(c.ids.begin(), c.ids.end()).size() == k, "repeated edge");
        for (int i = 0; i < k; i++) {
            auto ed = e[c.ids[i]];
            int a = c.verts[i], b = c.verts[(i + 1) % k];
            CHECK((ed.first == a && ed.second == b) || (ed.first == b && ed.second == a), "edge " << c.ids[i] << " does not join consecutive vertices");
        }
        // directed
        auto d = g.digraph(n, g.rnd(0, 12), g.rnd(0, 2) == 0, g.rnd(0, 1));
        if (g.rnd(0, 1)) d = g.dag(n, g.rnd(0, 12));
        g_case = dump_edges(n, d) + " directed";
        auto adj = to_adj(n, d, true);
        auto sc = strict_closure(n, d);
        bool cyc = false;
        for (int v = 0; v < n; v++) cyc |= sc[v][v] == 1;
        auto dc = find_cycle_directed(adj);
        CHECK(!dc.empty() == cyc, "directed found=" << !dc.empty() << " expected " << cyc);
        k = dc.size();
        CHECK((int)set<int>(dc.begin(), dc.end()).size() == k, "repeated vertex (directed)");
        for (int i = 0; i < k; i++) CHECK(has_arc(adj, dc[i], dc[(i + 1) % k]), "missing arc in directed cycle");
    }
    report_ok(cases);
}

// test_dsu.cpp -- DSU against naive labels (on a merge, relabel every element of one set).
#define TEST_NAME "dsu"
#include "check.hpp"
#include "../lib/dsu.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 3000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 30), q = g.rnd(1, 80);
        DSU d(n);
        vector<int> lab(n);
        iota(lab.begin(), lab.end(), 0);
        int sets = n;
        g_case = "n=" + to_string(n) + " ops:";
        for (int i = 0; i < q; i++) {
            int a = g.rnd(0, n - 1), b = g.rnd(0, n - 1), op = g.rnd(0, 2);
            g_case += string(" ") + (op ? "U" : "Q") + to_string(a) + "," + to_string(b);
            if (op) {
                bool merged = lab[a] != lab[b];
                if (merged) { int old = lab[b]; for (auto& x : lab) if (x == old) x = lab[a]; sets--; }
                CHECK(d.unite(a, b) == merged, "unite return value");
            }
            CHECK(d.same(a, b) == (lab[a] == lab[b]), "same()");
            CHECK(d.size(a) == (int)count(lab.begin(), lab.end(), lab[a]), "size()");
            CHECK(d.sets == sets, "sets");
        }
        for (int a = 0; a < n; a++)
            for (int b = 0; b < n; b++) CHECK(d.same(a, b) == (lab[a] == lab[b]), "final same()");
    }
    report_ok(cases);
}

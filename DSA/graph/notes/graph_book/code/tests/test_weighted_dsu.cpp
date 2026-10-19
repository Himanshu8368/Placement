// test_weighted_dsu.cpp -- potential DSU against a naive model: explicit labels and explicit
// potentials, shifting the whole merged group on every successful unite.
#define TEST_NAME "weighted_dsu"
#include "check.hpp"
#include "../lib/weighted_dsu.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 3000; seed++, cases++) {
        Gen g(seed);
        int n = g.rnd(1, 20), q = g.rnd(1, 60);
        WeightedDSU d(n);
        vector<int> lab(n);
        iota(lab.begin(), lab.end(), 0);
        vector<long long> pot(n, 0);
        g_case = "n=" + to_string(n) + " ops:";
        for (int i = 0; i < q; i++) {
            int a = g.rnd(0, n - 1), b = g.rnd(0, n - 1);
            long long w = g.rnd(-5, 5);
            // half of the unites are consistent with what is known, to exercise "true" on same set
            if (lab[a] == lab[b] && g.rnd(0, 1)) w = pot[b] - pot[a];
            g_case += " U" + to_string(a) + "," + to_string(b) + "," + to_string(w);
            bool expect;
            if (lab[a] == lab[b]) expect = pot[b] - pot[a] == w;
            else {
                expect = true;
                long long shift = pot[a] + w - pot[b];      // move b's group so pot[b] = pot[a] + w
                int old = lab[b];
                for (int x = 0; x < n; x++)
                    if (lab[x] == old) { lab[x] = lab[a]; pot[x] += shift; }
            }
            CHECK(d.unite(a, b, w) == expect, "unite return value");
            int x = g.rnd(0, n - 1), y = g.rnd(0, n - 1);
            CHECK(d.same(x, y) == (lab[x] == lab[y]), "same()");
            if (lab[x] == lab[y]) CHECK(d.diff(x, y) == pot[y] - pot[x], "diff(" << x << "," << y << ")");
        }
    }
    report_ok(cases);
}

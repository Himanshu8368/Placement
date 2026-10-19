// test_grid_bfs.cpp -- GridBFS against Floyd-Warshall on the explicitly built cell graph
// (4 or 8 neighbours, several start cells); reconstructed paths are valid moves.
#define TEST_NAME "grid_bfs"
#include "check.hpp"
#include "../lib/grid_bfs.hpp"

int main() {
    int cases = 0;
    for (int seed = 1; seed <= 3000; seed++, cases++) {
        Gen g(seed);
        int R = g.rnd(1, 5), C = g.rnd(1, 5), k = g.rnd(1, 3);
        bool eight = g.rnd(0, 1);
        auto grid = g.grid(R, C, g.rnd(0, 50));
        vector<pair<int, int>> starts;
        for (int i = 0; i < k; i++) starts.push_back({(int)g.rnd(0, R - 1), (int)g.rnd(0, C - 1)});
        g_case = (eight ? "8-dir grid:" : "4-dir grid:");
        for (auto& row : grid) g_case += " " + row;
        g_case += " starts:";
        for (auto& s : starts) g_case += " (" + to_string(s.first) + "," + to_string(s.second) + ")";
        // explicit graph on cells
        int N = R * C;
        vector<pair<int, int>> e;
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++)
                for (int dr = -1; dr <= 1; dr++)
                    for (int dc = -1; dc <= 1; dc++) {
                        if ((dr == 0 && dc == 0) || (!eight && dr != 0 && dc != 0)) continue;
                        int nr = r + dr, nc = c + dc;
                        if (nr < 0 || nr >= R || nc < 0 || nc >= C) continue;
                        if (grid[r][c] == '#' || grid[nr][nc] == '#') continue;
                        e.push_back({r * C + c, nr * C + nc});
                    }
        auto D = apsp_unweighted(N, e, true);
        GridBFS gb;
        gb.run(grid, starts, eight);
        for (int r = 0; r < R; r++)
            for (int c = 0; c < C; c++) {
                int best = -1;
                for (auto& s : starts) {
                    if (grid[s.first][s.second] == '#') continue;
                    int d = D[s.first * C + s.second][r * C + c];
                    if (d != -1 && (best == -1 || d < best)) best = d;
                }
                CHECK(gb.dist[r][c] == best, "dist at (" << r << "," << c << ") = " << gb.dist[r][c] << ", expected " << best);
                auto p = gb.path(r, c);
                if (best == -1) { CHECK(p.empty(), "path to unreachable cell"); continue; }
                CHECK((int)p.size() == best + 1 && p.back() == make_pair(r, c), "path length");
                for (size_t i = 0; i + 1 < p.size(); i++) {
                    int a = abs(p[i].first - p[i + 1].first), b = abs(p[i].second - p[i + 1].second);
                    CHECK(max(a, b) == 1 && (eight || a + b == 1) && grid[p[i + 1].first][p[i + 1].second] != '#', "bad move");
                }
            }
    }
    report_ok(cases);
}

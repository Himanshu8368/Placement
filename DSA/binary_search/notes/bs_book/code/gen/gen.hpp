#pragma once
// gen.hpp -- seeded random generators for testing array code.
//   Gen g(seed);                       same seed -> same arrays (reproducible failures)
//   g.rnd(l, r)                        uniform integer in [l, r]
//   g.array(n, lo, hi)                 n values uniform in [lo, hi]
//   g.sorted(n, lo, hi)                the same, sorted non-decreasingly
//   g.distinct(n, lo, hi)              n distinct values in [lo, hi] (needs hi-lo+1 >= n)
//   g.perm(n, base)                    random permutation of base..base+n-1
//   g.few_values(n, k)                 n values from a set of k random values (many duplicates)
//   g.intervals(n, lo, hi)             n pairs (l, r) with lo <= l <= r <= hi
//   g.matrix(r, c, lo, hi)             r x c matrix of values in [lo, hi]
//   print_array(a)                     "n" newline then the values (contest format)
// Small value ranges are the best bug finders: they create duplicates, zeros and ties.
#include <bits/stdc++.h>
using namespace std;

struct Gen {
    mt19937_64 rng;
    explicit Gen(unsigned long long seed) : rng(seed) {}
    long long rnd(long long l, long long r) { return uniform_int_distribution<long long>(l, r)(rng); }

    vector<long long> array(int n, long long lo, long long hi) {
        vector<long long> a(n);
        for (auto &x : a) x = rnd(lo, hi);
        return a;
    }
    vector<long long> sorted(int n, long long lo, long long hi) {
        vector<long long> a = array(n, lo, hi);
        sort(a.begin(), a.end());
        return a;
    }
    vector<long long> distinct(int n, long long lo, long long hi) {
        set<long long> s;
        while ((int)s.size() < n) s.insert(rnd(lo, hi));
        vector<long long> a(s.begin(), s.end());
        shuffle(a.begin(), a.end(), rng);
        return a;
    }
    vector<int> perm(int n, int base = 1) {
        vector<int> p(n);
        iota(p.begin(), p.end(), base);
        shuffle(p.begin(), p.end(), rng);
        return p;
    }
    vector<long long> few_values(int n, int k) {
        vector<long long> pool = array(k, -5, 5), a(n);
        for (auto &x : a) x = pool[rnd(0, k - 1)];
        return a;
    }
    vector<pair<long long, long long>> intervals(int n, long long lo, long long hi) {
        vector<pair<long long, long long>> v(n);
        for (auto &p : v) {
            long long x = rnd(lo, hi), y = rnd(lo, hi);
            p = make_pair(min(x, y), max(x, y));
        }
        return v;
    }
    vector<vector<long long>> matrix(int r, int c, long long lo, long long hi) {
        vector<vector<long long>> m(r);
        for (auto &row : m) row = array(c, lo, hi);
        return m;
    }
};

template <class T>
void print_array(const vector<T> &a) {
    printf("%d\n", (int)a.size());
    for (size_t i = 0; i < a.size(); i++) printf("%lld%c", (long long)a[i], i + 1 == a.size() ? '\n' : ' ');
}

"""Step frames for the front half of one-dimensional DP.

Drawing lives elsewhere. ``stories`` returns data only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, line_step, row_step, run_call, show

DAYS = [1, 4, 6, 7, 8, 20]
COSTS = [2, 7, 15]
DUR = (1, 7, 30)
JUMP = [1, -1, -2, 4, -7, 3]
JUMP_K = 2
ST = [1, 2, 3, 3]
EN = [3, 4, 5, 6]
PR = [50, 10, 40, 70]


def _seen(values, known):
    return [values[i] if i in known else "·" for i in range(len(values))]


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    if not 1 <= len(steps) <= 8:
        raise AssertionError((title, len(steps)))
    return (title, inp, steps, text)


def _jobs():
    idx = sorted(range(len(ST)), key=lambda i: (EN[i], i))
    jobs = [(ST[i], EN[i], PR[i]) for i in idx]
    ends = [e for _, e, _ in jobs]
    return jobs, ends


def _p_scan(jobs, i):
    """How many of the first i jobs end at or before the start of job i (1-based)."""
    start = jobs[i - 1][0]
    p = 0
    while p < i and jobs[p][1] <= start:
        p += 1
    return p


def _tickets_rec():
    def build(log):
        days, costs, dur = DAYS, COSTS, DUR
        n = len(days)

        def f(i):
            log.call(f"f({i})")
            if i >= n:
                return log.base(0, f"f({i}) = 0")
            best, how = 10**18, ""
            for p in range(3):
                j = i
                while j < n and days[j] < days[i] + dur[p]:
                    j += 1
                cand = costs[p] + f(j)
                if cand < best:
                    best, how = cand, f"{costs[p]} + f({j})"
            return log.ret(best, f"f({i}) = {how}")

        return f(0)

    ans, steps = run_call(build)
    assert ans == 11
    return _pack("Minimum cost for tickets — recursion", "days [1,4,6,7,8,20], costs [2,7,15]", steps, ans)


def _tickets_memo():
    def build(log):
        days, costs, dur = DAYS, COSTS, DUR
        n = len(days)
        memo = [None] * (n + 1)

        def f(i):
            if i >= n:
                log.call(f"f({i})")
                return log.base(0, f"f({i}) = 0")
            if memo[i] is not None:
                return log.hit(f"f({i})", memo[i])
            log.call(f"f({i})")
            best, how = 10**18, ""
            for p in range(3):
                j = i
                while j < n and days[j] < days[i] + dur[p]:
                    j += 1
                cand = costs[p] + f(j)
                if cand < best:
                    best, how = cand, f"{costs[p]} + f({j})"
            memo[i] = best
            return log.ret(best, f"f({i}) = {how}")

        return f(0)

    ans, steps = run_call(build)
    assert ans == 11
    return _pack("Minimum cost for tickets — memo", "days [1,4,6,7,8,20], costs [2,7,15]", steps, ans)


def _tickets_day():
    days, costs = DAYS, COSTS
    last = days[-1]
    travel = set(days)
    dp = [0] * (last + 1)
    steps = []
    shown = [0, 1, 4, 6, 7, 8, 20]

    def snap(title, hi_day, eq):
        known = {d for d in range(hi_day + 1)}
        cells = [dp[d] if d in known else "·" for d in shown]
        hi = shown.index(hi_day) if hi_day in shown else None
        steps.append(row_step(title, [("day", shown, hi), ("dp", cells, hi)], eq))

    snap("dp[0] = 0", 0, "nothing to cover")
    for d in range(1, last + 1):
        if d not in travel:
            dp[d] = dp[d - 1]
            if d == 19:
                snap("days 9..19 copy 9", 8, f"dp[19] = dp[8] = {show(dp[8])}")
            continue
        dp[d] = min(
            dp[d - 1] + costs[0],
            dp[max(0, d - 7)] + costs[1],
            dp[max(0, d - 30)] + costs[2],
        )
        if d in (1, 4, 6, 7, 8, 20):
            snap(
                f"day {d} → {show(dp[d])}",
                d,
                f"{show(dp[d])} = min({show(dp[d - 1])}+2, {show(dp[max(0, d - 7)])}+7, {show(dp[max(0, d - 30)])}+15)",
            )
    assert dp[last] == 11 and len(steps) <= 8
    return _pack("Minimum cost for tickets — day table", "days [1,4,6,7,8,20], costs [2,7,15]", steps, dp[last])


def _tickets_event():
    days, costs = DAYS, COSTS
    n = len(days)
    E = [0] * (n + 1)
    j7 = j30 = 0
    known = {0}
    steps = [row_step("E[0] = 0", [("E", _seen(E, known), 0)], "no trips yet")]
    for i in range(1, n + 1):
        d = days[i - 1]
        while days[j7] <= d - 7:
            j7 += 1
        while days[j30] <= d - 30:
            j30 += 1
        E[i] = min(E[i - 1] + costs[0], E[j7] + costs[1], E[j30] + costs[2])
        known.add(i)
        steps.append(row_step(
            f"trip {i - 1} day {d} → {show(E[i])}",
            [("E", _seen(E, known), i)],
            f"min(E[{i - 1}]+2, E[{j7}]+7, E[{j30}]+15) = {show(E[i])}",
        ))
    assert E[n] == 11 and len(steps) <= 8
    return _pack("Minimum cost for tickets — travel days", "days [1,4,6,7,8,20], costs [2,7,15]", steps, E[n])


def _jump_rec():
    def build(log):
        a, k = JUMP, JUMP_K

        def f(i):
            log.call(f"f({i})")
            if i == 0:
                return log.base(a[0], f"f(0) = {show(a[0])}")
            best, src = -10**18, 0
            for j in range(max(0, i - k), i):
                cand = f(j)
                if cand > best:
                    best, src = cand, j
            return log.ret(a[i] + best, f"f({i}) = {show(a[i])} + f({src})")

        return f(len(a) - 1)

    ans, steps = run_call(build)
    assert ans == 7
    return _pack("Jump game VI — recursion", "nums [1,-1,-2,4,-7,3], k = 2", steps, ans)


def _jump_memo():
    def build(log):
        a, k = JUMP, JUMP_K
        memo = [None] * len(a)

        def f(i):
            if i == 0:
                log.call("f(0)")
                return log.base(a[0], f"f(0) = {show(a[0])}")
            if memo[i] is not None:
                return log.hit(f"f({i})", memo[i])
            log.call(f"f({i})")
            best, src = -10**18, 0
            for j in range(max(0, i - k), i):
                cand = f(j)
                if cand > best:
                    best, src = cand, j
            memo[i] = a[i] + best
            return log.ret(memo[i], f"f({i}) = {show(a[i])} + f({src})")

        return f(len(a) - 1)

    ans, steps = run_call(build)
    assert ans == 7
    return _pack("Jump game VI — memo", "nums [1,-1,-2,4,-7,3], k = 2", steps, ans)


def _jump_heap():
    import heapq

    a, k = JUMP, JUMP_K
    n = len(a)
    dp = [0] * n
    heap = []
    steps = []
    dp[0] = a[0]
    heapq.heappush(heap, (-dp[0], 0))
    known = {0}
    steps.append(row_step("start", [("dp", _seen(dp, known), 0)], f"push (dp=1, j=0)"))
    for i in range(1, n):
        while heap and heap[0][1] < i - k:
            heapq.heappop(heap)
        best_j = heap[0][1]
        dp[i] = a[i] + dp[best_j]
        known.add(i)
        steps.append(row_step(
            f"i = {i}",
            [("dp", _seen(dp, known), i)],
            f"{show(dp[i])} = {show(a[i])} + dp[{best_j}]",
        ))
        heapq.heappush(heap, (-dp[i], i))
    assert dp[-1] == 7 and len(steps) <= 8
    return _pack("Jump game VI — heap", "nums [1,-1,-2,4,-7,3], k = 2", steps, dp[-1])


def _jump_deque():
    a, k = JUMP, JUMP_K
    n = len(a)
    dp = [0] * n
    dq = []
    steps = []
    dp[0] = a[0]
    dq.append(0)
    known = {0}
    steps.append(row_step(
        "start",
        [("dp", _seen(dp, known), 0), ("deque", [0], 0)],
        "dp[0] = 1",
    ))
    for i in range(1, n):
        while dq and dq[0] < i - k:
            dq.pop(0)
        dp[i] = a[i] + dp[dq[0]]
        known.add(i)
        front = dq[0]
        while dq and dp[dq[-1]] <= dp[i]:
            dq.pop()
        dq.append(i)
        steps.append(row_step(
            f"i = {i}",
            [("dp", _seen(dp, known), i), ("deque", list(dq), len(dq) - 1)],
            f"{show(dp[i])} = {show(a[i])} + dp[{front}]",
        ))
    assert dp[-1] == 7 and len(steps) <= 8
    return _pack("Jump game VI — deque", "nums [1,-1,-2,4,-7,3], k = 2", steps, dp[-1])


def _sched_rec():
    def build(log):
        jobs, _ends = _jobs()

        def f(i):
            log.call(f"f({i})")
            if i == 0:
                return log.base(0, "f(0) = 0")
            skip = f(i - 1)
            p = _p_scan(jobs, i)
            take = jobs[i - 1][2] + f(p)
            return log.ret(max(skip, take), f"f({i}) = max(f({i - 1}), {show(jobs[i - 1][2])} + f({p}))")

        return f(len(jobs))

    ans, steps = run_call(build)
    assert ans == 120
    return _pack(
        "Job scheduling — recursion",
        "start [1,2,3,3], end [3,4,5,6], profit [50,10,40,70]",
        steps,
        ans,
    )


def _sched_memo():
    def build(log):
        jobs, _ends = _jobs()
        memo = [None] * (len(jobs) + 1)

        def f(i):
            if i == 0:
                log.call("f(0)")
                return log.base(0, "f(0) = 0")
            if memo[i] is not None:
                return log.hit(f"f({i})", memo[i])
            log.call(f"f({i})")
            skip = f(i - 1)
            p = _p_scan(jobs, i)
            take = jobs[i - 1][2] + f(p)
            memo[i] = max(skip, take)
            return log.ret(memo[i], f"f({i}) = max(f({i - 1}), {show(jobs[i - 1][2])} + f({p}))")

        return f(len(jobs))

    ans, steps = run_call(build)
    assert ans == 120
    return _pack(
        "Job scheduling — memo",
        "start [1,2,3,3], end [3,4,5,6], profit [50,10,40,70]",
        steps,
        ans,
    )


def _sched_table():
    jobs, ends = _jobs()
    n = len(jobs)
    dp = [0] * (n + 1)
    known = {0}
    steps = [row_step("dp[0] = 0", [("dp", _seen(dp, known), 0), ("ends", ends, None)], "sorted by end")]
    for i in range(1, n + 1):
        p = _p_scan(jobs, i)
        w = jobs[i - 1][2]
        dp[i] = max(dp[i - 1], w + dp[p])
        known.add(i)
        steps.append(row_step(
            f"job {i}",
            [("dp", _seen(dp, known), i)],
            f"p = {p} by scan, max({show(dp[i - 1])}, {show(w)} + dp[{p}]) = {show(dp[i])}",
        ))
    assert dp[n] == 120 and len(steps) <= 8
    return _pack(
        "Job scheduling — table",
        "start [1,2,3,3], end [3,4,5,6], profit [50,10,40,70]",
        steps,
        dp[n],
    )


def _sched_bin():
    import bisect

    jobs, ends = _jobs()
    n = len(jobs)
    dp = [0] * (n + 1)
    known = {0}
    steps = [row_step("ends sorted", [("ends", ends, None), ("dp", _seen(dp, known), 0)], "dp[0] = 0")]
    for i in range(1, n + 1):
        p = bisect.bisect_right(ends, jobs[i - 1][0])
        w = jobs[i - 1][2]
        dp[i] = max(dp[i - 1], w + dp[p])
        known.add(i)
        steps.append(row_step(
            f"start {show(jobs[i - 1][0])}",
            [("dp", _seen(dp, known), i)],
            f"bisect_right(ends, {show(jobs[i - 1][0])}) = {p}, dp[{i}] = {show(dp[i])}",
        ))
    assert dp[n] == 120 and len(steps) <= 8
    return _pack(
        "Job scheduling — binary search",
        "start [1,2,3,3], end [3,4,5,6], profit [50,10,40,70]",
        steps,
        dp[n],
    )


def _houses_rec():
    def build(log):
        def ways(i):
            log.call(f"c({i})")
            if i == 0:
                return log.base(1, "c(0) = 1")
            if i == 1:
                return log.base(2, "c(1) = 2")
            return log.ret((ways(i - 1) + ways(i - 2)), f"c({i}) = c({i - 1}) + c({i - 2})")

        one = ways(1)
        return one * one

    ans, steps = run_call(build)
    steps.append(line_step("two sides", [f"{show(2)} × {show(2)} = {show(ans)}"], "the sides are independent"))
    assert ans == 4 and len(steps) <= 8
    return _pack("Place houses — recursion", "n = 1", steps, ans)


def _houses_memo():
    def build(log):
        memo = [None] * 2

        def ways(i):
            if i == 0:
                log.call("c(0)")
                return log.base(1, "c(0) = 1")
            if i == 1:
                log.call("c(1)")
                return log.base(2, "c(1) = 2")
            if memo[i] is not None:
                return log.hit(f"c({i})", memo[i])
            log.call(f"c({i})")
            memo[i] = ways(i - 1) + ways(i - 2)
            return log.ret(memo[i], f"c({i}) = c({i - 1}) + c({i - 2})")

        one = ways(1)
        return one * one

    ans, steps = run_call(build)
    steps.append(line_step("two sides", ["4 = 2 × 2"], "square the one-side count"))
    assert ans == 4 and len(steps) <= 8
    return _pack("Place houses — memo", "n = 1", steps, ans)


def _houses_table():
    c = [1, 2]
    steps = [
        row_step("c[0] = 1", [("c", ["1", "·"], 0)], "empty street"),
        row_step("c[1] = 2", [("c", ["1", "2"], 1)], "empty, or one house"),
        line_step("square", ["4 = 2 × 2"], "answer = c[1]²"),
    ]
    assert c[1] * c[1] == 4
    return _pack("Place houses — table", "n = 1", steps, c[1] * c[1])


def _houses_sq():
    prev2, prev1 = 1, 1
    steps = [chip_step(
        "c[-1], c[0]",
        [("prev2", prev2, "amber"), ("prev1", prev1, "indigo")],
        "formal c[-1] = 1 so the loop has no special case",
    )]
    cur = (prev1 + prev2) % (10**9 + 7)
    prev2, prev1 = prev1, cur
    steps.append(chip_step(
        "i = 1",
        [("prev2", prev2, "amber"), ("prev1", prev1, "indigo")],
        f"{show(prev1)} = 1 + 1",
    ))
    ans = prev1 * prev1 % (10**9 + 7)
    steps.append(chip_step(
        "square",
        [("one side", prev1, "indigo"), ("answer", ans, "green")],
        f"{show(ans)} = {show(prev1)} × {show(prev1)}",
    ))
    assert ans == 4
    return _pack("Place houses — one side, squared", "n = 1", steps, ans)


def stories():
    return {
        ("tickets", "rec"): _tickets_rec(),
        ("tickets", "memo"): _tickets_memo(),
        ("tickets", "day"): _tickets_day(),
        ("tickets", "event"): _tickets_event(),
        ("jump6", "rec"): _jump_rec(),
        ("jump6", "memo"): _jump_memo(),
        ("jump6", "heap"): _jump_heap(),
        ("jump6", "deque"): _jump_deque(),
        ("sched", "rec"): _sched_rec(),
        ("sched", "memo"): _sched_memo(),
        ("sched", "tab"): _sched_table(),
        ("sched", "bin"): _sched_bin(),
        ("houses", "rec"): _houses_rec(),
        ("houses", "memo"): _houses_memo(),
        ("houses", "tab"): _houses_table(),
        ("houses", "sq"): _houses_sq(),
    }


if __name__ == "__main__":
    data = stories()
    expect = {
        "tickets": "11",
        "jump6": "7",
        "sched": "120",
        "houses": "4",
    }
    for key, (_title, _inp, steps, final) in data.items():
        assert expect[key[0]] == final, (key, final)
        assert 1 <= len(steps) <= 8, (key, len(steps))
        assert all("title" in s for s in steps)
    print("keys", len(data))
    for key in data:
        print(key, data[key][3], "frames", len(data[key][2]))

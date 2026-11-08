"""Step frames for vacation, cooldown, and stock with at most k transactions."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_log import chip_step, line_step, row_step, run_call, show


def _pack(title, inp, steps, answer):
    text = answer if isinstance(answer, str) else show(answer)
    return (title, inp, steps, text)


def _vacation_rec():
    days = [(1, 2, 5), (3, 1, 1)]

    def build(log):
        def f(i, prev):
            tag = "-" if prev < 0 else str(prev)
            log.call(f"d{i}|{tag}")
            if i == len(days):
                return log.base(0, f"d{i}|{tag} = 0")
            best, how = 0, ""
            first = True
            for j in range(3):
                if j == prev:
                    continue
                got = days[i][j] + f(i + 1, j)
                if first or got > best:
                    best, how, first = got, f"{days[i][j]} + {show(got - days[i][j])}", False
            return log.ret(best, f"d{i}|{tag} = {how}")

        return f(0, -1)

    ans, steps = run_call(build)
    return _pack("Vacation — recursion", "days = [(1, 2, 5), (3, 1, 1)]", steps, ans)


def _vacation_memo():
    days = [(1, 2, 5), (3, 1, 1)]

    def build(log):
        memo = {}

        def f(i, prev):
            tag = "-" if prev < 0 else str(prev)
            label = f"d{i}|{tag}"
            if i == len(days):
                log.call(label)
                return log.base(0, f"{label} = 0")
            if (i, prev) in memo:
                return log.hit(label, memo[(i, prev)])
            log.call(label)
            best = 0
            how = ""
            first = True
            for j in range(3):
                if j == prev:
                    continue
                got = days[i][j] + f(i + 1, j)
                if first or got > best:
                    best, how, first = got, f"{days[i][j]}+{show(got - days[i][j])}", False
            memo[(i, prev)] = best
            return log.ret(best, f"{label} = {how}")

        return f(0, -1)

    ans, steps = run_call(build)
    return _pack("Vacation — memo", "days = [(1, 2, 5), (3, 1, 1)]", steps, ans)


def _vacation_tab():
    days = [(1, 2, 5), (3, 1, 1)]
    dp = [list(days[0])]
    steps = [row_step("day 0", [("dp", dp[0], None)], "start from day 0")]
    for x, y, z in days[1:]:
        a, b, c = dp[-1]
        nxt = [x + max(b, c), y + max(a, c), z + max(a, b)]
        dp.append(nxt)
        steps.append(row_step(
            "day 1",
            [("dp", nxt, 0)],
            f"{nxt[0]}, {nxt[1]}, {nxt[2]} = {x}+max({b},{c}), {y}+max({a},{c}), {z}+max({a},{b})",
        ))
    ans = max(dp[-1])
    steps.append(chip_step("answer", [("max", ans, "green")], f"max = {ans}"))
    return _pack("Vacation — tabulation", "days = [(1, 2, 5), (3, 1, 1)]", steps, ans)


def _vacation_opt():
    days = [(1, 2, 5), (3, 1, 1)]
    a, b, c = days[0]
    steps = [chip_step("day 0", [("a", a, "amber"), ("b", b, "amber"), ("c", c, "indigo")], "a, b, c = 1, 2, 5")]
    for x, y, z in days[1:]:
        na, nb, nc = x + max(b, c), y + max(a, c), z + max(a, b)
        eq = f"{na}, {nb}, {nc} from the old triple ({a}, {b}, {c})"
        a, b, c = na, nb, nc
        steps.append(chip_step("day 1", [("a", a, "green"), ("b", b, "indigo"), ("c", c, "indigo")], eq))
    ans = max(a, b, c)
    steps.append(chip_step("answer", [("a", a, "green"), ("b", b, "muted"), ("c", c, "muted")], f"max(a, b, c) = {ans}"))
    return _pack("Vacation — optimal", "days = [(1, 2, 5), (3, 1, 1)]", steps, ans)


def _cool_go(prices, log, memo=None):
    n = len(prices)

    def go(i, holding, cooling):
        label = f"{i}.{int(holding)}{int(cooling)}"
        if i == n:
            log.call(label)
            return log.base(0, f"{label} = 0")
        key = (i, holding, cooling)
        if memo is not None and key in memo:
            return log.hit(label, memo[key])
        log.call(label)
        rest = go(i + 1, holding, False)
        best, how = rest, f"rest {show(rest)}"
        if holding:
            sold = prices[i] + go(i + 1, False, True)
            if sold > best:
                best, how = sold, f"sell {prices[i]}+{show(sold - prices[i])}"
        elif not cooling:
            bought = -prices[i] + go(i + 1, True, False)
            if bought > best:
                best, how = bought, f"buy {show(bought)}"
        if memo is not None:
            memo[key] = best
        return log.ret(best, f"{label} = {how}")

    return go(0, False, False)


def _cooldown_rec():
    prices = [1, 2]

    def build(log):
        return _cool_go(prices, log)

    ans, steps = run_call(build)
    return _pack("Cooldown — recursion", "prices = [1, 2]", steps, ans)


def _cooldown_memo():
    prices = [1, 2]

    def build(log):
        return _cool_go(prices, log, memo={})

    ans, steps = run_call(build)
    return _pack("Cooldown — memo", "prices = [1, 2]", steps, ans)


def _cooldown_tab():
    prices = [1, 2, 3, 0, 2]
    n = len(prices)
    dp = [[[0, 0] for _ in range(2)] for _ in range(n + 1)]
    shown = {n, n - 1, 0}
    steps = []

    def snap(i, eq):
        cells = [dp[i][h][c] for h in (0, 1) for c in (0, 1)]
        steps.append(row_step(f"day {i}", [("h/c", cells, 0 if i == 0 else None)], eq))

    snap(n, "past the last day, every state is 0")
    for i in range(n - 1, -1, -1):
        for holding in (0, 1):
            for cooling in (0, 1):
                best = dp[i + 1][holding][0]
                if holding:
                    best = max(best, prices[i] + dp[i + 1][0][1])
                elif not cooling:
                    best = max(best, -prices[i] + dp[i + 1][1][0])
                dp[i][holding][cooling] = best
        if i in shown:
            snap(i, f"price {prices[i]}" if i else f"dp[0][0][0] = {dp[0][0][0]}")
    return _pack("Cooldown — tabulation", "prices = [1, 2, 3, 0, 2]", steps, dp[0][0][0])


def _cooldown_opt():
    prices = [1, 2, 3, 0, 2]
    hold, sold, rest = -math.inf, -math.inf, 0
    steps = [chip_step(
        "start",
        [("hold", hold, "amber"), ("sold", sold, "amber"), ("rest", rest, "indigo")],
        "hold and sold start at −∞, rest at 0",
    )]
    for i, p_ in enumerate(prices):
        nh, ns, nr = max(hold, rest - p_), hold + p_, max(rest, sold)
        eq = f"day {i}, price {p_}: rest uses the old sold"
        hold, sold, rest = nh, ns, nr
        last = i == len(prices) - 1
        steps.append(chip_step(
            f"price {p_}",
            [("hold", hold, "indigo"), ("sold", sold, "green" if last else "indigo"), ("rest", rest, "indigo")],
            eq,
        ))
    ans = max(sold, rest)
    return _pack("Cooldown — optimal", "prices = [1, 2, 3, 0, 2]", steps, ans)


def _stock_go(k, prices, log, memo=None):
    n = len(prices)

    def go(i, left, holding):
        label = f"{i}.{left}{int(holding)}"
        if i == n:
            log.call(label)
            return log.base(0, f"{label} = 0")
        key = (i, left, holding)
        if memo is not None and key in memo:
            return log.hit(label, memo[key])
        log.call(label)
        best = go(i + 1, left, holding)
        how = f"rest {show(best)}"
        if holding:
            sold = prices[i] + go(i + 1, left, False)
            if sold > best:
                best, how = sold, f"sell {show(sold)}"
        elif left:
            bought = -prices[i] + go(i + 1, left - 1, True)
            if bought > best:
                best, how = bought, f"buy {show(bought)}"
        if memo is not None:
            memo[key] = best
        return log.ret(best, f"{label} = {how}")

    return go(0, k, False)


def _stock_rec():
    def build(log):
        return _stock_go(2, [2, 4, 1], log)

    ans, steps = run_call(build)
    return _pack("Stock k — recursion", "k = 2, prices = [2, 4, 1]", steps, ans)


def _stock_memo():
    def build(log):
        return _stock_go(2, [2, 4, 1], log, memo={})

    ans, steps = run_call(build)
    return _pack("Stock k — memo", "k = 2, prices = [2, 4, 1]", steps, ans)


def _stock_tab():
    k, prices = 2, [2, 4, 1]
    n = len(prices)
    dp = [[[0, 0] for _ in range(k + 1)] for _ in range(n + 1)]
    steps = [line_step("day 3", ["every state past the last price is 0"], "filled from the end")]
    for i in range(n - 1, -1, -1):
        for left in range(k + 1):
            for holding in (0, 1):
                best = dp[i + 1][left][holding]
                if holding:
                    best = max(best, prices[i] + dp[i + 1][left][0])
                elif left:
                    best = max(best, -prices[i] + dp[i + 1][left - 1][1])
                dp[i][left][holding] = best
        steps.append(row_step(
            f"price {prices[i]}",
            [("sell left", [dp[i][j][0] for j in range(k + 1)], k)],
            f"after price {prices[i]}, not holding, buys left 0..{k}",
        ))
    return _pack("Stock k — tabulation", "k = 2, prices = [2, 4, 1]", steps, dp[0][k][0])


def _stock_opt1():
    k, prices = 2, [2, 4, 1]
    buy = [-math.inf] * (k + 1)
    sell = [0] * (k + 1)
    steps = [chip_step(
        "start",
        [("buy1", buy[1], "amber"), ("sell1", sell[1], "indigo"), ("buy2", buy[2], "amber"), ("sell2", sell[2], "indigo")],
        "buy starts at −∞, sell at 0",
    )]
    for p_ in prices:
        for j in range(1, k + 1):
            buy[j] = max(buy[j], sell[j - 1] - p_)
            sell[j] = max(sell[j], buy[j] + p_)
        steps.append(chip_step(
            f"price {p_}",
            [("buy1", buy[1], "indigo"), ("sell1", sell[1], "green"), ("buy2", buy[2], "indigo"), ("sell2", sell[2], "green")],
            f"j upward; an empty trade at price {p_} cannot beat a real one",
        ))
    return _pack("Stock k — optimal 1", "k = 2, prices = [2, 4, 1]", steps, max(sell))


def _stock_opt2():
    k, prices = 2, [2, 4, 1]
    rises = [b - a for a, b in zip(prices, prices[1:])]
    kept = [max(0, d) for d in rises]
    steps = [
        line_step("cap", [f"k = {k}, n//2 = {len(prices) // 2}", "k is at least n//2, so the cap never binds"], ""),
        chip_step("day rises", [("4−2", rises[0], "green"), ("1−4", rises[1], "amber")], "keep a rise only when it is positive"),
        chip_step("sum", [("answer", sum(kept), "green")], f"{kept[0]} + {kept[1]} = {sum(kept)}"),
    ]
    return _pack("Stock k — optimal 2", "k = 2, prices = [2, 4, 1]", steps, sum(kept))


def stories():
    return {
        ("vacation", "rec"): _vacation_rec(),
        ("vacation", "memo"): _vacation_memo(),
        ("vacation", "tab"): _vacation_tab(),
        ("vacation", "opt"): _vacation_opt(),
        ("cooldown", "rec"): _cooldown_rec(),
        ("cooldown", "memo"): _cooldown_memo(),
        ("cooldown", "tab"): _cooldown_tab(),
        ("cooldown", "opt"): _cooldown_opt(),
        ("stock", "rec"): _stock_rec(),
        ("stock", "memo"): _stock_memo(),
        ("stock", "tab"): _stock_tab(),
        ("stock", "opt1"): _stock_opt1(),
        ("stock", "opt2"): _stock_opt2(),
    }


if __name__ == "__main__":
    board = stories()
    expect = {
        ("vacation", "rec"): "8", ("vacation", "memo"): "8", ("vacation", "tab"): "8", ("vacation", "opt"): "8",
        ("cooldown", "rec"): "1", ("cooldown", "memo"): "1", ("cooldown", "tab"): "3", ("cooldown", "opt"): "3",
        ("stock", "rec"): "2", ("stock", "memo"): "2", ("stock", "tab"): "2", ("stock", "opt1"): "2", ("stock", "opt2"): "2",
    }
    for key, final in expect.items():
        assert board[key][3] == final, (key, board[key][3])
        assert 3 <= len(board[key][2]) <= 8, (key, len(board[key][2]))
        print(key, board[key][3])
    print("PASS")

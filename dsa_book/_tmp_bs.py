import random


def mo_move(qs, B):
    order = sorted(range(len(qs)), key=lambda i: (qs[i][0] // B, qs[i][1]))
    move = 0
    prev = (0, -1)
    for i in order:
        l, r = qs[i]
        move += abs(l - prev[0]) + abs(r - prev[1])
        prev = (l, r)
    return move


rm = random.Random(7)
tot_good = tot_sqrt = 0
for _ in range(40):
    n = rm.choice([1000, 2000, 4000])
    q = rm.choice([100, 200, 400])       # q much smaller than n
    qs = [tuple(sorted((rm.randrange(n), rm.randrange(n)))) for _ in range(q)]
    b_good = max(1, int(n / (q ** 0.5)))
    b_sqrt = max(1, int(round(n ** 0.5)))
    tot_good += mo_move(qs, b_good)
    tot_sqrt += mo_move(qs, b_sqrt)
print("n/sqrt(q) total:", tot_good, " sqrt(n) total:", tot_sqrt, " ratio:", tot_good / tot_sqrt)

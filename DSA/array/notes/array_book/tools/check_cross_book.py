"""check_cross_book.py -- no problem may be used both in this book and in a companion book
(DP, graph, tree, string, recursion, stack-queue-heap). Compares the problem URLs in this book's chapters with
every URL in the companions' chapters and every problem planned in their outlines. Run from the array_book folder."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import companions  # noqa: E402

slugs = companions.lc_slug_map()
mine = {companions.key_of_url(u, slugs): u for u in companions.chapter_urls(os.path.join(companions.HERE, '..'))}
used = companions.used_keys(slugs)
shared = 0
for book, keys in used.items():
    for k in sorted(set(mine) & keys):
        shared += 1
        print('ALSO IN THE %s BOOK: %s' % (book.upper(), mine[k]))
print('%d array-book problems, %d shared with a companion book' % (len(mine), shared))

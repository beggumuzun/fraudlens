"""Union-find (disjoint set union), written from scratch.

Answers "are these two things connected, directly or through a chain?" in near-constant
time. Here the things are cards and devices: a card paid from a device links them, and a
fraud ring shows up as one big group of cards all chained together through its devices.

Two classic optimisations make it fast:

* path compression - after finding a root, point every node on the way straight at it
* union by size    - always hang the smaller tree under the larger one

Together they make each operation run in amortised O(alpha(n)) time, where alpha is the
inverse Ackermann function: less than 5 for any input that fits in the universe.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable


class UnionFind:
    def __init__(self, items: Iterable[Hashable] = ()) -> None:
        self._parent: dict[Hashable, Hashable] = {}
        self._size: dict[Hashable, int] = {}
        for item in items:
            self.add(item)

    def __len__(self) -> int:
        return len(self._parent)

    def __contains__(self, item: Hashable) -> bool:
        return item in self._parent

    def add(self, item: Hashable) -> None:
        if item not in self._parent:
            self._parent[item] = item
            self._size[item] = 1

    def find(self, item: Hashable) -> Hashable:
        """The representative (root) of the item's group, compressing the path on the way."""
        self.add(item)
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:  # second pass: point everything at the root
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, a: Hashable, b: Hashable) -> Hashable:
        """Merge the groups of a and b. Returns the root of the merged group."""
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return root_a
        if self._size[root_a] < self._size[root_b]:
            root_a, root_b = root_b, root_a
        self._parent[root_b] = root_a
        self._size[root_a] += self._size[root_b]
        return root_a

    def connected(self, a: Hashable, b: Hashable) -> bool:
        return self.find(a) == self.find(b)

    def groups(self) -> list[list[Hashable]]:
        """Every group, as a list of its members."""
        members: dict[Hashable, list[Hashable]] = defaultdict(list)
        for item in self._parent:
            members[self.find(item)].append(item)
        return list(members.values())

from pysat.solvers import Minisat22
import random
from functools import cached_property
from typing import Protocol, Annotated, Literal, Optional
import math
import matplotlib.pyplot as plt
import numpy as np
from pydantic import BaseModel, Field
from timerun import Timer

MAX_SKILL = 10


class Player(BaseModel):
    id_: int
    name: str
    skill: Annotated[int, Field(ge=1, le=10)]  # higher is better
    gender: Literal["m", "f"]
    joker_for: Optional["Player"] = None

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = f"{self.id_}{self.gender}"
        if self.joker_for:
            s += f" J{self.joker_for.id_}"
        return s

    def as_joker(self, id_: int) -> "Player":
        player = Player(
            id_=id_,
            name=f"{self.name}-J",
            skill=self.skill,
            gender=self.gender,
            joker_for=self
        )
        self.joker_for = player
        return player


def create_players(num_players: int, *, rand: bool = True, jokers: int = 0) -> list[Player]:
    random.seed(44)
    players = [Player(id_=idx, name=f"Player {idx}", skill=random.randint(1, MAX_SKILL) if rand else idx + 1, gender="m" if idx % 2 == 0 else "f") for
               idx in range(num_players)]
    max_id = max(players, key=lambda player: player.id_).id_ + 1
    if jokers > 0:
        cpy = players.copy()
        random.shuffle(cpy)
        for i in range(jokers):
            players.append(
                cpy[i].as_joker(max_id + i)
            )
    return players


class ProbabilitySet(dict[int, float]):

    def remove(self, id_: int) -> None:
        if id_ not in self:
            raise KeyError()

        removed_prob = self.pop(id_)
        new_norm = 1 - removed_prob

        for idx, prob in self.copy().items():
            self[idx] = self[idx] / new_norm

        assert np.isclose(sum(self.values()), 1)

    def pick(self):
        selection = np.random.choice(np.array(list(self.keys())), size=1, p=np.array(list(self.values())))
        idx = int(selection[0])
        self.remove(idx)
        return idx


class PlayerMetricsProvider(Protocol):

    def get_metrics(self, players: list[Player]) -> ProbabilitySet:
        """
        Returns for each player a metric
        :param players:
        """


class NormalizedSkillMetricsProvider(PlayerMetricsProvider):

    def get_metrics(self, players: list[Player]) -> ProbabilitySet:
        overall = 1.0 * sum(p.skill for p in players)
        return ProbabilitySet({p.id_: (1.0 * p.skill) / overall for p in players})


class Pair:
    def __init__(self, player1: Player, player2: Player, joker_pair: bool = False):
        self.player1 = player1
        self.player2 = player2
        self._joker_pair = joker_pair

    def value(self):
        s1 = self.player1.skill - MAX_SKILL / 2.0
        s2 = self.player2.skill - MAX_SKILL / 2.0

        a = abs(s1 + s2 - 1) / MAX_SKILL

        return 1 - a
        # return 1 - (1.0 * abs(self.player1.skill + self.player2.skill) / (1.0 * MAX_SKILL))

    @cached_property
    def players(self):
        return {self.player1.id_, self.player2.id_}

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f"Pair({self.player1}, {self.player2}: {self.value():.2f})"

    @property
    def id(self):
        return self.player1.id_, self.player2.id_

    def has_player_joker_conflict(self, p: Player) -> bool:
        if p.joker_for is None:
            return False
        return p.joker_for.id_ in self.players

    def has_joker_conflict(self, pair: "Pair") -> bool:
        return self.has_player_joker_conflict(pair.player1) or self.has_player_joker_conflict(pair.player2)


class PairFilter(Protocol):

    def is_valid(self, p1: Player, p2: Player) -> bool:
        return not (p1.joker_for == p2 and p2.joker_for == p1)


class EmptyFilter(PairFilter):
    def is_valid(self, p1: Player, p2: Player) -> bool:
        return PairFilter.is_valid(self, p1, p2)


class MixedFilter(PairFilter):
    def is_valid(self, p1: Player, p2: Player) -> bool:
        return PairFilter.is_valid(self, p1, p2) and p1.gender != p2.gender


class SameGenderFilter(PairFilter):
    def is_valid(self, p1: Player, p2: Player) -> bool:
        return PairFilter.is_valid(self, p1, p2) and p1.gender == p2.gender


class Filter:
    same: PairFilter = SameGenderFilter()
    default: PairFilter = EmptyFilter()
    mixed: PairFilter = MixedFilter()


class PairSet:
    def __init__(self, players: list[Player], filter_strategy: PairFilter | None = None):
        if not filter_strategy:
            filter_strategy = Filter.default
        self._original_pairs = [Pair(p1, p2) for p1 in players for p2 in players if p1.id_ < p2.id_ and filter_strategy.is_valid(p1, p2)]
        self._pairs = set(self._original_pairs)

    def __iter__(self):
        return iter(self._pairs)

    def remove(self, pair: Pair) -> None:

        to_remove = []
        for p in self._pairs:
            if p.players.intersection(pair.players):
                to_remove.append(p)

        for p in to_remove:
            self._pairs.remove(p)

    def can_pick(self):
        return len(self._pairs) > 0

    def pick(self) -> Pair:
        ranking = list(x.value() for x in self._pairs)
        total_sum = 1.0 * sum(ranking)
        probabilities = list(1.0 * x.value() / total_sum for x in self._pairs)
        selection = np.random.choice(np.array(list(self._pairs)), size=1, p=np.array(probabilities))
        selected_pair = selection[0]
        self.remove(selected_pair)
        return selected_pair

    def sample(self, n: int = 1):
        pairs = []
        prob = 1
        original_pairs = self._pairs.copy()
        for i in range(n):
            if not self.can_pick():
                break
            pair = self.pick()
            pairs.append(pair)
            prob *= pair.value()
        self._pairs = original_pairs
        if len(pairs) != n:
            return None
        return prob, pairs


def draw(pairs: PairSet):
    min_idx = min(min(p.player1.id_, p.player2.id_) for p in pairs)
    max_idx = max(max(p.player1.id_, p.player2.id_) for p in pairs)

    x_vals = range(min_idx, max_idx + 1)
    y_vals = range(min_idx, max_idx + 1)

    # Initialize heatmap array
    heatmap = np.zeros((len(y_vals), len(x_vals)))

    # Fill heatmap array
    for p in pairs:
        heatmap[p.player2.id_, p.player1.id_] = p.value()

    # Plot heatmap
    plt.imshow(heatmap, cmap='viridis', origin='lower')
    plt.colorbar(label='Value')
    plt.xticks(ticks=range(len(x_vals)), labels=x_vals)
    plt.yticks(ticks=range(len(y_vals)), labels=y_vals)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Heatmap of Pair Values')
    plt.savefig('heatmap.png')


class TopN:

    def __init__(self, n: int):
        self._items = []
        self._max = n

    def append(self, item, value: int):
        self._items.append((item, value))
        self._items = sorted(self._items, key=lambda x: x[1], reverse=True)[:self._max]

    @property
    def items(self):
        return self._items


def _joker_overlap(pair1: Pair, pair2: Pair):
    return (
            pair1.has_joker_conflict(pair2) or
            pair2.has_joker_conflict(pair1)  # redundant
    )


def _solve_matching(edges: set[tuple[int, int]]):
    nodes = {n for n1, n2 in edges for n in [n1, n2]}
    edge_map = {edge: idx + 1 for idx, edge in enumerate(edges)}
    inverse_edge_map = {idx: edge for edge, idx in edge_map.items()}

    node_edge_map = {node: [(n1, n2) for n1, n2 in edges if n1 == node or n2 == node] for node in nodes}

    with Minisat22() as m:
        for node, edges in node_edge_map.items():
            # at most one
            for e1, e2 in [(e1, e2) for e1 in edges for e2 in edges if edge_map[e1] < edge_map[e2]]:
                edge1_id = edge_map[e1]
                edge2_id = edge_map[e2]
                m.add_clause([-edge1_id, -edge2_id])
            # at least one
            m.add_clause([edge_map[edge] for edge in edges])

        r = m.solve()
        if r:
            model = m.get_model()
            selected_edges = [inverse_edge_map[edge_idx] for edge_idx in model if edge_idx > 0]
            return selected_edges
        return None


def get_matches(pairs: list[Pair]):
    edges = set()
    inverse_pair_map = {idx: pair for idx, pair in enumerate(pairs)}
    pair_map = {pair.id: idx for idx, pair in enumerate(pairs)}

    for pair in pairs:
        for other_pair in [p for p in pairs if p != pair]:
            p1_id = pair_map[pair.id]
            p2_id = pair_map[other_pair.id]
            if not _joker_overlap(pair, other_pair):
                lower = min(p1_id, p2_id)
                upper = max(p1_id, p2_id)
                edges.add((lower, upper))

    with Timer() as t:
        solution = _solve_matching(edges)
    # print(f"Solving took {t.duration.nanoseconds / 1.e6:.3f} ms")

    if solution is None:
        return None

    matches = [(inverse_pair_map[pair1], inverse_pair_map[pair2]) for pair1, pair2 in solution]
    return matches


def run():
    num_samples = 1000
    num_top = 50
    number_rounds = 10
    num_players = 11
    num_games_per_round = math.ceil(num_players / 4)
    num_jokers = 4 * num_games_per_round - num_players
    players = create_players(num_players=num_players, rand=True, jokers=num_jokers)  # add a secondary player for the jokers

    s = PairSet(players, filter_strategy=Filter.default)
    # filter such that the jokers cannot play with themselves
    # add a filter such that constraints can be implemented (MM, FF, FM, unrestricted) +++

    for round_idx in range(number_rounds):
        # add solver for pairs with constraints
        top_n = TopN(num_top)
        with Timer() as t:
            for i in range(num_samples):
                if sample := s.sample(num_games_per_round * 2):
                    prob, pairs = sample
                    if matches := get_matches(pairs):
                        top_n.append(matches, prob)

        selected_pairs, selected_prob = random.choice(top_n.items)

        print(f"Round {round_idx + 1}:")
        print(f"\tSolving took: {t.duration.nanoseconds / 1.e9:.3f}s - Quality: {selected_prob:.4f}")
        print("\tMatches:")
        for idx, (p1, p2) in enumerate(selected_pairs):
            print(f"\t\tMatch {idx + 1}:\t[{p1}] vs [{p2}] ==> {prob:.3f}")
        print("---------------------------------------------------------------")


if __name__ == "__main__":
    run()

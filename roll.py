import heapq
import operator
import random
import re
import types
import typing

numeric = typing.Union[int, float]


class Roll:
    def __init__(
        self,
        qty: int,
        die: int,
        adv: bool,
        disadv: bool,
        keep: int,
        modifiers: typing.List[Modifier],
        operation: types.FunctionType,
    ):
        self.qty = qty
        self.die = die
        self.adv = adv
        self.disadv = disadv
        self.keep = keep
        self.modifiers = modifiers
        self.operation = operation

        if self.qty > 1000:
            raise ValueError("Maximum quantity exceeded.")

        self.resolved = False
        self._rolls = []
        self._total = 0

    def __str__(self):
        return self.desc_str() + self.roll_str()

    def __repr__(self):
        return str(self)

    def full_str(self, pad_desc: int = 0, pad_roll: int = 0) -> str:
        return self.desc_str(pad_desc) + self.roll_str(pad_roll)

    def desc_str(self, pad_to: int = 0) -> str:
        if (self.adv or self.disadv) and (self.qty == 2):
            string = f"d{self.die}"
        else:
            string = f"{self.qty if self.qty > 1 else ''}d{self.die}"
        string += f"{'a' if self.adv else ''}{'d' if self.disadv else ''}"
        if self.keep >= 0:
            string += f" keep {self.keep}"
        for mod in self.modifiers:
            string += f" {str(mod)}"

        while len(string) < pad_to:
            string += " "

        return string

    def dice_str(self) -> str:
        # Used for saving results in database
        return f"{self.qty}d{self.die}"

    def roll_str(self, pad_to: int = 0) -> str:
        string = ""
        if len(self.rolls) > 1:
            string += f"\tRolls: {str(self.rolls)[1:-1]}"
            string += f" \tTotal: {clean_number(self.total)}"
        else:
            string += f"\tRoll: {self.rolls[0]}"
            if self.modifiers:
                string += f" \tTotal: {clean_number(self.total)}"

        while len(string) < pad_to:
            string += " "

        return string

    @property
    def total(self) -> numeric:
        if not self.resolved:
            self.resolve()
        return self._total

    @property
    def rolls(self) -> typing.List[int]:
        if not self.resolved:
            self.resolve()
        return self._rolls

    def ensure_rolled(self) -> None:
        if self._rolls is None:
            self._rolls = [random.randint(1, self.die) for _ in range(self.qty)]

    def calculate_total(self) -> numeric:
        if self.adv:
            total = max(self._rolls)
        elif self.disadv:
            total = min(self._rolls)
        elif self.keep >= 0:
            total = sum(heapq.nlargest(self.keep, self._rolls))
        else:
            total = sum(self._rolls)

        self._total = self.apply_mods(total)
        return self._total

    def resolve(self) -> (typing.List[int], numeric):
        self.ensure_rolled()
        self.calculate_total()
        self.resolved = True

        return self._rolls, self._total

    def reroll(self, n: int) -> None:
        for r in heapq.nsmallest(n, self.rolls):
            self._rolls[self._rolls.index(r)] = random.randint(1, self.die)

        return self._rolls, self.calculate_total()

    def apply(self, val: numeric) -> numeric:
        return self.operation(val, self.total)

    def apply_mods(self, val: numeric) -> numeric:
        for mod in self.modifiers:
            val = mod.apply(val)
        return val

    def set_operator(self, opstr: str) -> None:
        self.operation = get_operator(opstr)

    def add_modifier(self, modifier: Modifier) -> None:
        self.modifiers.append(modifier)


class Modifier:
    def __init__(self, val: numeric, opstr: str):
        self.val = val
        self.opstr = opstr
        self.operation = get_operator(opstr)

    def __str__(self):
        return f"{self.opstr} {self.val}"

    def apply(self, val: numeric) -> numeric:
        return self.operation(val, self.val)

    @staticmethod
    def from_string(string: str) -> Modifier:
        return Modifier(int(string[1:]), string[0])


def get_operator(opstr: str) -> types.FunctionType:
    return {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
    }[opstr]


def clean_number(num: numeric) -> numeric:
    if num // 1 == num:
        return int(num)
    return round(num, 2)


def get_rolls(string: str, max_qty: int=-1) -> typing.List[Roll]:
    rolls = []
    regex = re.compile(
        r"(?P<val>\d+(?= *[+-/*]))? *(?P<op>[+-/*]?)? *(?P<n>\d+"
        r"(?= +))? *(?P<qty>\d*)d(?P<die>\d+) *(?P<advstr>[ad]*)"
        r"(?P<mods>( *(k *\d+|[+-/*] *\d+(?!d)))*(?![\dd]))"
    )

    count = 0
    for roll in re.finditer(regex, string):
        n = roll.group("n")
        if n in [None, ""]:
            n = 1
        else:
            n = int(n)

        qty = roll.group("qty")
        if qty in [None, ""]:
            qty = 1
        else:
            qty = int(qty)

        if qty == 0:
            raise ValueError("I can't roll a zero sided die.")

        die = int(roll.group("die"))

        advstr = roll.group("advstr")
        advscore = advstr.count("a") - advstr.count("d")
        adv = advscore > 0
        disadv = advscore < 0
        if (adv or disadv) and qty == 1:
            qty = 2

        modstr = roll.group("mods")
        mods = []
        keep = -1
        if not modstr in [None, ""]:
            modstr = modstr.replace(" ", "") + "k"
            # Add k to ensure last term is added
            # won't cause an error in a properly formatted string

            q = ""
            o = ""
            for c in modstr:
                if c in ["k", "+", "-", "*", "/"]:
                    if o == "k" and q:
                        if keep > 0:
                            raise ValueError("Multiple keep values provided.")
                        keep = int(q)
                    elif o:
                        mods.append(Modifier(int(q), o))
                    o = c
                    q = ""
                else:
                    q += c

        op = roll.group("op")
        if op in [None, ""] or n > 1:
            opstr = "+"
        else:
            opstr = op

        val = roll.group("val")
        if not val in [None, ""]:
            mods.insert(0, Modifier(int(val), op))
            op = get_operator("+")
        else:
            op = get_operator(opstr)

        for _ in range(n):
            rolls.append(Roll(qty, die, adv, disadv, keep, mods, op))

        count += 1
        if max_qty > 0 and count == max_qty:
            break 

    return rolls

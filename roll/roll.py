import heapq
import numbers
import operator
import random
import re
from sre_constants import MAX_REPEAT
import typing


class Modifier:
    """Stores a value and operation and can apply itself to another value."""

    def __init__(self, val: numbers.Number, opstr: str):
        self.val = val
        self.opstr = opstr
        self.operation = get_operator(opstr)

    def __str__(self):
        return f"{self.opstr} {self.val}"

    def apply(self, val: numbers.Number) -> numbers.Number:
        return self.operation(val, self.val)

    @staticmethod
    def from_string(string: str) -> "Modifier":
        return Modifier(int(string[1:]), string[0])


class Expr:
    def __init__(self, modifiers: typing.List[Modifier], opstr: str) -> None:
        self.modifiers = modifiers
        self.opstr = opstr
        self.operation = get_operator(opstr)

        self._total = None

    def __str__(self):
        return self.desc_str()

    def __repr__(self):
        return f"<{self.__class__.__name__}:{str(self)}>"

    @property
    def total(self) -> numbers.Number:
        if self._total is None:
            self.resolve()
        return self._total

    def full_str(self, pad_desc: int = 0, pad_val: int = 0) -> str:
        string = (
            self.desc_str(pad_desc)
            + "\t"
            + self.val_str(pad_val)
            + f"\tTotal: {clean_number(self.total)}"
        )

        return string

    def desc_str(self, pad_to, include_op):
        raise NotImplementedError()

    def val_str(self):
        raise NotImplementedError()

    def roll_str(self):
        return self.val_str()

    def str_prefix(self, include_opstr: bool = False):
        return (self.opstr + " ") if include_opstr else ""

    def modf_str(self, include_opstr: bool = False):
        return self.str_prefix(include_opstr) + (
            ("{} " + " ".join(str(mod) for mod in self.modifiers))
            if self.modifiers
            else "{}"
        )

    def resolve(self) -> None:
        self.calculate_total()

    def calculate_total(self):
        raise NotImplementedError(
            "Abstract Expr class should not be directly instantiated."
        )

    def apply(self, val: numbers.Number) -> numbers.Number:
        return self.operation(val, self.total)

    def apply_mods(self, val: numbers.Number) -> numbers.Number:
        for mod in self.modifiers:
            val = mod.apply(val)
        return val

    def set_operator(self, opstr: str) -> None:
        self.operation = get_operator(opstr)

    def add_modifier(self, modifier: Modifier) -> None:
        self.modifiers.append(modifier)


class SuperExpr(Expr):
    def __init__(
        self,
        exprs: typing.List["Expr"],
        modifiers: typing.List[Modifier],
        opstr: str,
    ) -> None:
        super().__init__(modifiers, opstr)
        self.exprs = exprs

    def desc_str(self, pad_to: int = 0, include_opstr: bool = False):
        return (
            self.modf_str(include_opstr)
            .format(
                "("
                + " ".join(
                    e.desc_str(include_opstr=(i != 0))
                    for (i, e) in enumerate(self.exprs)
                )
                + ")"
            )
            .ljust(pad_to)
        )

    def val_str(self, pad_to: int = 0, include_opstr: bool = False):
        return (
            self.str_prefix()
            + "("
            + " ".join(
                e.val_str(include_opstr=(i != 0))
                for (i, e) in enumerate(self.exprs)
            )
            + ")"
        ).ljust(pad_to)

    def calculate_total(self):
        self._total = self.apply_mods(calculate_total(self.exprs))

    def clone(self):
        return SuperExpr(
            [e.clone() for e in self.exprs], self.modifiers[:], self.opstr
        )


class ConstExpr(Expr):
    def __init__(
        self,
        val: int,
        modifiers: typing.List[Modifier],
        opstr: str,
    ) -> None:
        super().__init__(modifiers, opstr)
        self.val = val

    def desc_str(self, pad_to: int = 0, include_opstr: bool = False):
        return self.modf_str(include_opstr).format(self.val_str()).ljust(pad_to)

    def val_str(self, pad_to: int = 0, include_opstr: bool = False):
        return (self.str_prefix(include_opstr) + str(self.val)).ljust(pad_to)

    def calculate_total(self):
        self._total = self.apply_mods(self.val)

    def clone(self):
        return ConstExpr(self.val, self.modifiers[:], self.opstr)


class RollExpr(Expr):
    """Stores information and returns results for a parsed roll."""

    MAX_QTY = 1000

    def __init__(
        self,
        modifiers: typing.List[Modifier],
        opstr: str,
        qty: int,
        die: int,
        adv: bool,
        disadv: bool,
        keep: int,
    ):
        super().__init__(modifiers, opstr)
        self.qty = qty
        self.die = die
        self.adv = adv
        self.disadv = disadv
        self.keep = keep

        if self.qty > RollExpr.MAX_QTY:
            raise ValueError("Maximum quantity exceeded.")

        self._rolls = None

    @property
    def rolls(self):
        if self._rolls is None:
            self.resolve()
        return self._rolls

    def full_str(self, pad_desc: int = 0, pad_val: int = 0) -> str:
        return (
            self.desc_str(pad_desc, False)
            + "\t"
            + self.roll_str(pad_val)
            + f"\tTotal: {clean_number(self.total)}"
        )

    def desc_str(self, pad_to: int = 0, include_opstr=False) -> str:
        string = "d" + str(self.die)

        is_adv_roll = (self.adv or self.disadv) and (self.qty == 2)
        if self.qty > 1 and not is_adv_roll:
            string = str(self.qty) + string

        if self.adv:
            string += "a"
        if self.disadv:
            string += "d"

        if self.keep >= 0:
            string += f" keep {self.keep}"

        return self.modf_str(include_opstr).format(string).ljust(pad_to)

    def dice_str(self) -> str:
        # Used for saving results in database
        return f"{self.qty}d{self.die}"

    def val_str(self, pad_to: int = 0, include_opstr: bool = False):
        return (
            self.str_prefix(include_opstr)
            + ", ".join([str(r) for r in self.rolls])
        ).ljust(pad_to)

    def roll_str(self, pad_to: int = 0) -> str:
        if len(self.rolls) > 1:
            string = f"Rolls: {str(self.rolls)[1:-1]}"
        else:
            string = f"Roll: {self.rolls[0]}"

        while len(string) < pad_to:
            string += " "

        return string

    def calculate_total(self) -> numbers.Number:
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

    def ensure_rolled(self) -> None:
        if self._rolls is None:
            self._rolls = [random.randint(1, self.die) for _ in range(self.qty)]

    def resolve(self) -> None:
        self.ensure_rolled()
        self.calculate_total()

    def reroll(self, n: int) -> typing.Tuple[typing.List[int], numbers.Number]:
        for r in heapq.nsmallest(n, self.rolls):
            self._rolls[self._rolls.index(r)] = random.randint(1, self.die)

        return self._rolls, self.calculate_total()

    def clone(self):
        return RollExpr(
            self.modifiers[:],
            self.opstr,
            self.qty,
            self.die,
            self.adv,
            self.disadv,
            self.keep,
        )


def get_operator(opstr: str) -> typing.Callable:
    """Return a function related to a given numbers.Number operation string."""

    # For the inverse division operator (hack), division is simply called
    # with the operands reversed. Pylint assumes that this is a mistake when
    # it is in fact intended. Same thing for subtraction and inverse.
    #
    # pylint: disable=arguments-out-of-order

    return {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "\\": lambda a, b: operator.truediv(b, a),
        "~": lambda a, b: operator.sub(b, a),
    }[opstr]


def clean_number(num: numbers.Number) -> numbers.Number:
    """Round to 2 or less decimal places."""

    if num // 1 == num:
        return int(num)
    return round(num, 2)


MODS_REGEX = re.compile(r"(?P<op>[+-/*]) *(?P<const>\d+)")


def parse_mods(modstr: str) -> typing.List[Modifier]:
    """Parse a string of modifier statements."""

    if not modstr:
        return []

    mods = []
    for mod in re.finditer(MODS_REGEX, modstr):
        mods.append(Modifier(int(mod.group("const")), mod.group("op")))

    return mods


CONST = r"(?P<const>\d+(?= *[+-/*]))"
OP = r"(?P<op>[+-/*])"
N = r"(?P<n>\d+(?= +))"
EXPR = r"\((?P<expr>.*)\)"
ROLL = r"(?P<qty>\d*)d(?P<die>\d+)(?P<advstr>[ad]*)?(k *(?P<keep>\d+))?"
C = r"(?P<c>\d+)"
MODS = r"(?P<mods>( *[+-/*] *\d+(?!d))*(?![\dd]))"
EXPR_REGEX = re.compile(rf"{CONST}? *({OP}|{N})? *({EXPR}|{ROLL}|{C}) *{MODS}?")


def empty_group(match_group):
    return match_group is None or match_group == ""


def parse_integer_group(match_group, default=None):
    if empty_group(match_group):
        return default
    else:
        return int(match_group)


def parse_roll_expr(expr) -> RollExpr:
    qty = parse_integer_group(expr.group("qty"), 1)

    if qty == 0:
        raise ValueError("Can't roll a zero sided die.")

    die = parse_integer_group(expr.group("die"))

    advstr = expr.group("advstr")
    adv = "a" in advstr
    disadv = "d" in advstr

    # if under advantage and disadvantage, roll is straight
    if adv and disadv:
        adv = disadv = False

    # a roll like d20a implicitly means 2d20a
    if (adv or disadv) and qty == 1:
        qty = 2

    keep = parse_integer_group(expr.group("keep"), -1)

    return qty, die, adv, disadv, keep


def parse_expr(string: str, max_qty: int = None):
    exprs = []
    for match in re.finditer(EXPR_REGEX, string.lower()):

        const = parse_integer_group(match.group("const"))
        opstr = "+" if empty_group(match.group("op")) else match.group("op")
        n = parse_integer_group(match.group("n"), 1)
        mods = parse_mods(match.group("mods"))

        if not empty_group(match.group("expr")):
            expr = SuperExpr(parse_expr(match.group("expr")), mods, opstr)
        elif not empty_group(match.group("c")):
            expr = ConstExpr(parse_integer_group(match.group("c")), mods, opstr)
        else:
            expr = RollExpr(mods, opstr, *parse_roll_expr(match))

        if const is not None:
            # other operations are commutative
            if opstr == "/":
                opstr = "\\"
            elif opstr == "-":
                opstr = "~"

            expr.opstr = "+"
            special = expr.clone()
            special.modifiers.insert(0, Modifier(const, opstr))
            exprs.append(special)
            n -= 1

        for _ in range(n):
            exprs.append(expr.clone())

    return exprs[:max_qty]


def get_rolls(string: str, max_qty: int = None):
    return parse_expr(string, max_qty)


def calculate_total(exprs: typing.List[Expr]) -> numbers.Number:
    """Calculate to total result of a list of Expr objects."""

    total = 0
    for expr in exprs:
        total = expr.apply(total)

    return total


def get_result(string: str) -> numbers.Number:
    """Calculate and return the total of rolls parsed from a string."""

    return calculate_total(get_rolls(string))


def rolls_string(rolls: typing.List[Expr]) -> str:
    """Return a descriptive string for a list of Roll objects."""

    if len(rolls) == 1:
        return str(rolls[0])
    elif rolls:
        pad_desc = max([len(r.desc_str()) for r in rolls])
        pad_roll = max([len(r.roll_str()) for r in rolls])
        message = "\n".join(r.full_str(pad_desc, pad_roll) for r in rolls)
        message += "\nGrand Total: " + str(calculate_total(rolls))

        return message
    else:
        return ""

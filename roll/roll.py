import heapq
import operator
import random
import re
import typing

Number = typing.Union[int, float]


class Modifier:
    """Stores a value and operation and can apply itself to another value."""

    def __init__(self, val: Number, opstr: str):
        self.val = val
        self.opstr = opstr
        self.operation = get_operator(opstr)

    def __str__(self):
        return f"{self.opstr} {self.val}"

    def apply(self, val: Number) -> Number:
        return self.operation(val, self.val)

    @staticmethod
    def from_string(string: str) -> "Modifier":
        return Modifier(int(string[1:]), string[0])


class Expr:
    """Base expression class, inherited by specific types."""

    def __init__(self, modifiers: typing.List[Modifier], opstr: str) -> None:
        self.modifiers = modifiers
        self.set_opstr(opstr)

        self._total: typing.Optional[Number] = None

    def __str__(self):
        return self.desc_str()

    def __repr__(self):
        return f"<{self.__class__.__name__}:{str(self)}>"

    @property
    def total(self) -> Number:
        if self._total is None:
            self.resolve()

        # after calling resolve this is always non-None, but mypy doesn't
        # know this.
        return self._total  # type: ignore

    def roll_info(self) -> typing.List[typing.Tuple[str, int]]:
        return []

    def set_opstr(self, opstr):
        self.opstr = opstr
        self.operation = get_operator(opstr)

    def full_str(self, pad_desc: int = 0, pad_val: int = 0) -> str:
        string = (
            self.desc_str(pad_desc)
            + "\t"
            + self.val_str(pad_val)
            + f"\tTotal: {clean_number(self.total)}"
        )

        return string

    def desc_str(self, pad_to: int = 0, include_opstr: bool = False):
        raise NotImplementedError()

    def val_str(self, pad_to: int = 0, include_opstr: bool = False):
        raise NotImplementedError()

    def roll_str(self, pad_to: int = 0):
        return self.val_str(pad_to)

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

    def apply(self, val: Number) -> Number:
        return self.operation(val, self.total)

    def apply_mods(self, val: Number) -> Number:
        for mod in self.modifiers:
            val = mod.apply(val)
        return val

    def set_operator(self, opstr: str) -> None:
        self.operation = get_operator(opstr)

    def add_modifier(self, modifier: Modifier) -> None:
        self.modifiers.append(modifier)


class SuperExpr(Expr):
    """Expression term which contains other expressions as children."""

    def __init__(
        self,
        exprs: typing.List["Expr"],
        modifiers: typing.List[Modifier],
        opstr: str,
    ) -> None:
        super().__init__(modifiers, opstr)
        self.exprs = exprs

    def roll_info(self) -> typing.List[typing.Tuple[str, int]]:
        info = []
        for e in self.exprs:
            info.extend(e.roll_info())
        return info

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
            self.modf_str(include_opstr).format(
                "("
                + " ".join(
                    e.val_str(include_opstr=(i != 0))
                    for (i, e) in enumerate(self.exprs)
                )
                + ")"
            )
        ).ljust(pad_to)

    def calculate_total(self):
        self._total = self.apply_mods(calculate_total(self.exprs))

    def clone(self):
        return SuperExpr(
            [e.clone() for e in self.exprs], self.modifiers[:], self.opstr
        )


class ConstExpr(Expr):
    """Constant term in an expression, just an integer."""

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
        return (self.modf_str(include_opstr).format(str(self.val))).ljust(
            pad_to
        )

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

        self._rolls: typing.Optional[typing.List[int]] = None

    @property
    def rolls(self):
        if self._rolls is None:
            self.resolve()
        return self._rolls

    def roll_info(self) -> typing.List[typing.Tuple[str, int]]:
        return [(self.dice_str(), self.rolls)]

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
        roll = ", ".join([str(r) for r in self.rolls])

        if len(self.rolls) > 1:
            if self.adv:
                val = f"max({roll})"
            elif self.disadv:
                val = f"min({roll})"
            elif self.keep >= 0:
                val = f"({roll} k{self.keep})"
            else:
                val = f"sum({roll})"
        else:
            val = roll

        return self.modf_str(include_opstr).format(val).ljust(pad_to)

    def roll_str(self, pad_to: int = 0) -> str:
        if len(self.rolls) > 1:
            string = f"Rolls: {str(self.rolls)[1:-1]}"
        else:
            string = f"Roll: {self.rolls[0]}"

        return string.ljust(pad_to)

    def calculate_total(self) -> Number:
        if self._rolls is None:
            raise ValueError("Can't calculated total before rolling!")

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

    def reroll(self, n: int) -> typing.Tuple[typing.List[int], Number]:
        # accessing self.rolls means that self._rolls is not None, though
        # mypy doesn't know this.

        for r in heapq.nsmallest(n, self.rolls):

            self._rolls[self._rolls.index(r)] = random.randint(1, self.die)  # type: ignore

        return self._rolls, self.calculate_total()  # type:ignore

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
    """Return a function related to a given Number operation string."""

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


def clean_number(num: Number) -> Number:
    """Round to 2 or less decimal places."""

    if num // 1 == num:
        return int(num)
    return round(num, 2)


# Matches a valid roll modifier, like + 2
MODS_REGEX = re.compile(r"(?P<op>[+-/*]) *(?P<const>\d+)")


def parse_mods(modstr: str) -> typing.List[Modifier]:
    """Parse a string of modifier statements."""

    if not modstr:
        return []

    mods = []
    for mod in re.finditer(MODS_REGEX, modstr):
        mods.append(Modifier(int(mod.group("const")), mod.group("op")))

    return mods


# Used by extract_modstr to find the largest valid mod string.
VALID_MODS_REGEX = re.compile(rf"^( *{MODS_REGEX.pattern})*")


def extract_modstr(string: str) -> str:
    """Find the maximal substring that can be parsed as roll mods."""
    # This regex always matches at least the empty string, so .match(string) is
    # never None and .group(0) is always valid.
    return VALID_MODS_REGEX.match(string).group(0)  # type: ignore


def empty_group(match_group):
    """Check if a regex capture group is None or the empty string."""
    return match_group is None or match_group == ""


def parse_integer_group(match_group, default=None):
    """Handling for an integer capture group from a regular expression."""
    if empty_group(match_group):
        return default
    else:
        return int(match_group)


def parse_roll_expr(expr) -> typing.Tuple[int, int, bool, bool, int]:
    """Parse a die roll expression into the arguments need for a RollExpr."""
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


def parse_expr(string: str, opstr: str = "+", modstr: str = ""):
    """Parse a bracketed sub-expression in a roll expression."""
    this_expr = ""
    ambig = ""
    other_exprs = ""
    stack = 0
    expr_ended = False
    for i, c in enumerate(string):
        if c == "(":
            if expr_ended:
                other_exprs = string[i:]
                break
            else:
                if stack != 0:
                    this_expr += c
                stack += 1
        elif c == ")":
            stack -= 1
            if stack == 0:
                expr_ended = True
            else:
                this_expr += c
        else:
            if expr_ended:
                ambig += c
            else:
                this_expr += c

    this_modstr = extract_modstr(ambig)
    other_exprs = ambig[max(len(this_modstr) - 1, 0) :] + other_exprs
    if other_exprs:
        other_exprs += modstr
    else:
        this_modstr += modstr

    return [
        SuperExpr(parse(this_expr.strip()), parse_mods(this_modstr), opstr)
    ] + parse(other_exprs.strip())


# Regular expressions used in parsing. Broken up into more comprehensible
# sub sections. Overall structure of a roll expression is any number of
# expressions of the form
#
# expr :=
#   | <const> <op> <expr> <mods>
#   | <n> <expr> <mods>
#   | <op> <expr> <mods>
#   | <roll>
#   | <c>
CONST = r"(?P<const>\d+(?= *[+-/*]))"
OP = r"(?P<op>[+-/*])"
N = r"(?P<n>\d+(?= +))"
EXPR = r"(?P<expr>\(.*\))"
ROLL = r"(?P<qty>\d*)d(?P<die>\d+)(?P<advstr>[ad]*)?(k *(?P<keep>\d+))?"
C = r"(?P<c>\d+)"
MODS = r"(?P<mods>( *[+-/*] *\d+(?!d))*(?![\dd]))"
EXPR_REGEX = re.compile(rf"{CONST}? *({OP}|{N})? *({EXPR}|{ROLL}|{C}) *{MODS}?")


def parse(string: str):
    """Parse a string of roll expressions."""

    exprs = []
    for match in re.finditer(EXPR_REGEX, string.lower()):

        const = parse_integer_group(match.group("const"))
        opstr = "+" if empty_group(match.group("op")) else match.group("op")
        n = parse_integer_group(match.group("n"), 1)
        mods = parse_mods(match.group("mods"))

        extras = []

        if not empty_group(match.group("expr")):
            expr, *extras = parse_expr(
                match.group("expr"), opstr, match.group("mods")
            )
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

            expr.set_opstr("+")
            special = expr.clone()
            special.modifiers.insert(0, Modifier(const, opstr))
            exprs.append(special)
            n -= 1

        for _ in range(n):
            exprs.append(expr.clone())

        exprs.extend(extras)

    return exprs


def get_rolls(string: str, max_qty: int = None):
    """Parse the input string, returning up to max_qty rolls."""
    return parse(string)[:max_qty]


def calculate_total(exprs: typing.List[Expr]) -> Number:
    """Calculate to total result of a list of Expr objects."""

    total: Number = 0
    for expr in exprs:
        total = expr.apply(total)

    return total


def get_result(string: str) -> Number:
    """Calculate and return the total of rolls parsed from a string."""

    return calculate_total(get_rolls(string))


def rolls_string(rolls: typing.List[Expr]) -> str:
    """Return a descriptive string for a list of Roll objects."""

    if len(rolls) == 1:
        return rolls[0].full_str()
    elif rolls:
        pad_desc = max([len(r.desc_str()) for r in rolls])
        pad_roll = max([len(r.roll_str()) for r in rolls])
        message = "\n".join(r.full_str(pad_desc, pad_roll) for r in rolls)
        message += "\nGrand Total: " + str(calculate_total(rolls))

        return message
    else:
        return ""

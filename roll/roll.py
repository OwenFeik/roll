import enum
import functools
import heapq
import operator
import random
import typing

# Generic numeric base type
Number = typing.Union[int, float]

# Length 1 strings
Char = str

# During lexing this is used for return tuples of the form
# (finished_token, current_token)
TokenPair = typing.Tuple[typing.Optional["Token"], typing.Optional["Token"]]

# Type used to represent the result of a given roll
# (roll_format, results)
RollInfo = typing.Tuple[str, typing.List[int]]

# Default seperator for output strings.
SEP = "\t"


class Token:
    def __init__(self, token=""):
        self.token = token

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, Token)
            and type(o) == type(self)
            and o.token == self.token
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}<{str(self)}>"

    def __str__(self) -> str:
        return self.token

    def consume(self, c: Char) -> TokenPair:
        return self, Token.from_char(c)

    @staticmethod
    def from_char(c: Char) -> typing.Optional["Token"]:
        if c.isspace():
            return None
        elif c.isnumeric():
            return IntegerToken(c)
        elif c == ".":
            return DecimalToken(c)
        elif c == "d":
            return RollToken(c)
        else:
            return NonTerminalToken.from_char(c)


class TerminalToken(Token):
    def value(self) -> typing.Any:
        raise NotImplementedError()


class IntegerToken(TerminalToken):
    def __init__(self, token):
        super().__init__(token=token)
        assert not token or token.isnumeric()

    def consume(self, c: Char) -> TokenPair:
        if c.isnumeric():
            self.token += c
            return None, self
        elif c == ".":
            return None, DecimalToken(self.token + c)
        elif c == "d":
            return None, RollToken(self.token + c)
        else:
            return self, Token.from_char(c)

    def value(self) -> int:
        return int(self.token)


class DecimalToken(TerminalToken):
    def __init__(self, token):
        super().__init__(token=token)
        whole, dot, decimal = token.partition(".")
        assert not whole or whole.isnumeric()
        assert dot == "."
        assert not decimal or decimal.isnumeric()

    def consume(self, c: Char) -> TokenPair:
        if c.isnumeric():
            self.token += c
            return None, self
        else:
            return self, Token.from_char(c)

    def value(self) -> float:
        return float(self.token)


class RollToken(TerminalToken):
    SEPERATOR = "d"

    def __init__(self, token):
        super().__init__(token=token)
        qty, d, size = token.partition(RollToken.SEPERATOR)
        assert not qty or qty.isnumeric()
        assert d == "d"
        assert not size or size.isnumeric()

    @property
    def qty(self):
        qty = self.token.partition(RollToken.SEPERATOR)[0]

        if qty:
            return int(qty)
        else:
            # d8 implicity means 1d8
            return 1

    @property
    def size(self):
        return int(self.token.partition(RollToken.SEPERATOR)[2])

    def consume(self, c: Char) -> TokenPair:
        if c.isnumeric():
            self.token += c
            return None, self
        elif self.token == "d":
            # This was actually a disadvantage operator
            return OperatorToken(self.token), Token.from_char(c)
        else:
            try:
                self.qty
            except ValueError:
                raise ValueError(f"Invalid die roll: {self.token}")

            return self, Token.from_char(c)


class NonTerminalToken(Token):
    @staticmethod
    def from_char(c: Char) -> typing.Optional[Token]:
        if c == OpenExprToken.CHARACTER:
            return OpenExprToken()
        elif c == CloseExprToken.CHARACTER:
            return CloseExprToken()
        else:
            return OperatorToken.from_char(c)


class OpenExprToken(NonTerminalToken):
    CHARACTER = "("

    def __init__(self):
        super().__init__(token=OpenExprToken.CHARACTER)


class CloseExprToken(NonTerminalToken):
    CHARACTER = ")"

    def __init__(self):
        super().__init__(token=CloseExprToken.CHARACTER)


class OperatorToken(NonTerminalToken):
    OPERATORS = "+-*/^kads"

    def __init__(self, token):
        super().__init__(token=token)

    @property
    def opstr(self):
        return self.token

    @staticmethod
    def from_char(c: Char) -> typing.Optional[Token]:
        if c in OperatorToken.OPERATORS:
            return OperatorToken(c)
        else:
            raise ValueError(f"Invalid character: {c}")


class Expr:
    def __repr__(self) -> str:
        return f"{type(self).__name__}<{str(self)}>"

    def __str__(self):
        return str(self.total)

    @property
    def value(self) -> Number:
        raise NotImplementedError()

    @property
    def total(self) -> Number:
        value = self.value
        if value // 1 == value:
            return int(value)
        return round(value, 2)

    def expr_str(self):
        return ""

    def roll_str(self) -> str:
        rolls = [str(n) for _, rolls in self.roll_info() for n in rolls]

        return (
            "Roll" + ("s" if len(rolls) > 1 else "") + ": " + ", ".join(rolls)
        )

    def full_str(self, sep=SEP):
        if len(self.roll_info()) > 1:
            value_str = self.expr_str()
        else:
            value_str = self.roll_str()
        return f"{self}{sep}{value_str}{sep}Total: {self.total}"

    def has_rolls(self):
        return bool(len(self.roll_info()))

    def roll_info(self) -> typing.List[RollInfo]:
        return []

    def reroll(self, n: int) -> typing.Tuple[typing.List[int], Number]:
        # note: this uses a reference to some sub expressions rolls to reroll.
        # see comment in RollExpr.reroll_info for detail.

        info = self.roll_info()
        if not info:
            raise ValueError("Can't reroll an expression with no rolls.")

        return RollExpr.reroll_info(info[0], n)

    def clone(self) -> "Expr":
        raise NotImplementedError()


class TerminalExpr(Expr):
    # Cap out at largest 32 bit signed integer to minimise
    # potential for dos attack
    MAX_VALUE = 2 ** 31 - 1

    def __init__(self):
        super().__init__()

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, TerminalExpr)
            and type(o) == type(self)
            and o.value == self.value
        )


class ConstExpr(TerminalExpr):
    def __init__(self, value: Number):
        super().__init__()
        self._value = value

        if self._value > TerminalExpr.MAX_VALUE:
            raise ValueError("Maximum constant size exceeded.")

    @property
    def value(self) -> Number:
        return self._value

    def expr_str(self):
        return str(self.total)

    def clone(self) -> "ConstExpr":
        return ConstExpr(self.value)


class RollExpr(TerminalExpr):
    # Things can get quite slow for large numbers of rolls, and there's little
    # practical reason to roll more dice anyway. This makes it safe(r) to allow
    # public access to roll.
    MAX_QTY = 1000

    def __init__(self, qty: int, size: int):
        super().__init__()
        self.qty = qty
        self.size = size
        self._rolls: typing.Optional[typing.List[int]] = None

        if self.qty > RollExpr.MAX_QTY:
            raise ValueError("Maximum quantity exceeded.")
        if self.size * self.qty > TerminalExpr.MAX_VALUE:
            raise ValueError("Maximum roll value exceeded.")

    def __eq__(self, o: object) -> bool:
        return (
            type(o) == type(self)
            and isinstance(o, RollExpr)
            and o.qty == self.qty
            and o.size == self.size
        )

    def __str__(self) -> str:
        return f"{self.qty}{RollToken.SEPERATOR}{self.size}"

    @property
    def rolls(self) -> typing.List[int]:
        if self._rolls is None:
            self._rolls = [
                random.randint(1, self.size) for _ in range(self.qty)
            ]
        return self._rolls

    @property
    def value(self) -> Number:
        return sum(self.rolls)

    def expr_str(self, summed=True):
        if self.qty > 1:
            string = "(" + ", ".join(map(str, self.rolls)) + ")"
            if summed:
                return "sum" + string
            else:
                return string
        else:
            return str(self.rolls[0])

    def roll_info(self) -> typing.List[RollInfo]:
        return [(str(self), self.rolls)]

    def reroll(self, n: int) -> typing.Tuple[typing.List[int], Number]:
        # accessing self.rolls means that self._rolls is not None, though
        # mypy doesn't know this.

        for r in heapq.nsmallest(n, self.rolls):
            self._rolls[  # type: ignore
                self._rolls.index(r)  # type: ignore
            ] = random.randint(1, self.size)

        return self._rolls, self.total  # type: ignore

    def clone(self) -> "RollExpr":
        return RollExpr(self.qty, self.size)

    @staticmethod
    def reroll_info(
        info: RollInfo, n
    ) -> typing.Tuple[typing.List[int], Number]:
        # This is a little hack to allow expressions to reroll RollExprs from
        # anywhere. It works because rolls from info is a reference to the rolls
        # of the original roll, so by creating a new equivalent roll using these
        # rolls and rerolling, we're actually rerolling the original dice.

        roll_string, rolls = info
        roll = RollExpr.from_token(RollToken(roll_string))
        roll._rolls = rolls
        return roll.reroll(n)

    @staticmethod
    def from_token(token: RollToken):
        return RollExpr(token.qty, token.size)


class NonTerminalExpr(Expr):
    def __init__(self, exprs: typing.List[Expr]) -> None:
        super().__init__()
        self.exprs = exprs

    def __eq__(self, o: object) -> bool:
        return (
            type(o) == type(self)
            and isinstance(o, NonTerminalExpr)
            and o.exprs == self.exprs
        )

    def roll_info(self) -> typing.List[RollInfo]:
        return [info for expr in self.exprs for info in expr.roll_info()]


class SuperExpr(NonTerminalExpr):
    def __init__(self, exprs: typing.List[Expr]):
        super().__init__(exprs)

    def __repr__(self):
        return (
            type(self).__name__
            + "<"
            + " + ".join(f"({repr(expr)})" for expr in self.exprs)
            + ">"
        )

    def __str__(self):
        return " + ".join(f"({expr})" for expr in self.exprs)

    @property
    def value(self) -> Number:
        return sum(expr.value for expr in self.exprs)

    def expr_str(self):
        return "(" + " + ".join(expr.expr_str() for expr in self.exprs) + ")"

    def clone(self) -> "SuperExpr":
        return SuperExpr([expr.clone() for expr in self.exprs])


class Fixity(enum.Enum):
    INFIX = enum.auto()  # a op b
    PREFIX = enum.auto()  # op a
    POSTFIX = enum.auto()  # a op


class OpExprSpecification:
    MAX_PRECEDENCE = 0

    def __init__(
        self,
        opstr: str,
        precedence: int,
        fixity: Fixity,
        arguments: typing.List[typing.Type[Expr]],
        klasse: typing.Callable,
    ) -> None:
        self.opstr = opstr
        self.precedence = precedence
        self.fixity = fixity
        self.arguments = arguments
        self.klasse = klasse

        if fixity is Fixity.INFIX:
            assert len(arguments) == 2
            self.left, self.right = arguments
        else:
            assert len(arguments) == 1
            self.arg = arguments[0]

        # Keep track of the operator with the highest precedence so that we
        # can iterate up to it.
        if precedence > OpExprSpecification.MAX_PRECEDENCE:
            OpExprSpecification.MAX_PRECEDENCE = precedence

    def satisfied(
        self,
        a: typing.Union[Expr, OperatorToken, None],
        b: typing.Union[Expr, OperatorToken, None],
        c: typing.Union[Expr, OperatorToken, None],
    ) -> bool:
        if not isinstance(b, OperatorToken) or self.opstr != b.opstr:
            return False

        if self.fixity is Fixity.INFIX:
            return isinstance(a, self.left) and isinstance(c, self.right)
        elif self.fixity is Fixity.PREFIX:
            return isinstance(c, self.arg)
        elif self.fixity is Fixity.POSTFIX:
            return isinstance(a, self.arg)

    def instantiate(
        self,
        a: typing.Union[Expr, OperatorToken, None],
        b: typing.Union[Expr, OperatorToken, None],
        c: typing.Union[Expr, OperatorToken, None],
    ) -> typing.Tuple[typing.List[Expr], int]:
        # The typing ignores in this function are to suppress type issues which
        # are handled by the below assertion.

        assert self.satisfied(a, b, c)

        if self.fixity is Fixity.INFIX:
            return [self.klasse(a, c)], -1  # type: ignore
        elif self.fixity is Fixity.PREFIX:
            return [a, self.klasse(c)], 0  # type: ignore
        elif self.fixity is Fixity.POSTFIX:
            return [self.klasse(a), c], 0  # type: ignore


class OpExpr(NonTerminalExpr):
    def __init__(self, opstr: Char, left: Expr, right: Expr) -> None:
        super().__init__([left, right])
        self.opstr = opstr
        self.left = left
        self.right = right

        # some subclasses don't use this behaviour, so I just handle the
        # KeyError. A little inelegant but w/e
        try:
            self.operation: typing.Union[
                typing.Callable, None
            ] = OpExpr.get_operation(opstr)
        except KeyError:
            self.operation = None

    def __eq__(self, o: object) -> bool:
        return (
            super().__eq__(o)
            and isinstance(o, OpExpr)
            and o.opstr == self.opstr
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}<{repr(self.left)} {self.opstr}"
            f" {repr(self.right)}>"
        )

    def __str__(self) -> str:
        return f"{self.left} {self.opstr} {self.right}"

    @property
    def value(self) -> Number:
        assert self.operation is not None
        return self.operation(self.left.value, self.right.value)

    def expr_str(self):
        return f"{self.left.expr_str()} {self.opstr} {self.right.expr_str()}"

    def roll_info(self) -> typing.List[RollInfo]:
        return self.left.roll_info() + self.right.roll_info()

    def clone(self) -> "OpExpr":
        return OpExpr(self.opstr, self.left.clone(), self.right.clone())

    @staticmethod
    def get_operation(opstr) -> typing.Callable:
        return {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
            "^": operator.pow,
        }[opstr]


class KeepExpr(OpExpr):
    OPERATOR = "k"

    def __init__(self, roll, keep) -> None:
        super().__init__(KeepExpr.OPERATOR, roll, keep)

        assert isinstance(roll, RollExpr)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}<{repr(self.roll)}{KeepExpr.OPERATOR}"
            f"{repr(self.keep)}>"
        )

    def __str__(self) -> str:
        return f"{self.roll}{KeepExpr.OPERATOR}{self.keep}"

    @property
    def roll(self) -> RollExpr:
        # guaranteed to be RollExpr by assertion in __init__, but mypy doesn't
        # know this.
        return self.left  # type: ignore

    @property
    def keep(self) -> Expr:
        return self.right

    @property
    def value(self) -> Number:
        return sum(heapq.nlargest(int(self.keep.value), self.roll.rolls))

    def expr_str(self):
        return f"{self.roll.expr_str(False)}k{self.keep.expr_str()}"

    def clone(self) -> "KeepExpr":
        return KeepExpr(self.roll.clone(), self.keep.clone())


class UnaryExpr(NonTerminalExpr):
    def __init__(self, opstr: str, fixity: Fixity, expr: Expr) -> None:
        super().__init__([expr])

        self.opstr = opstr
        self.fixity = fixity
        self.expr = expr

        self.operation = UnaryExpr.get_operation(opstr)

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, UnaryExpr)
            and o.opstr == self.opstr
            and o.fixity is self.fixity
            and o.expr == self.expr
        )

    def __repr__(self) -> str:
        name = type(self).__name__
        if self.fixity == Fixity.PREFIX:
            return f"{name}<{self.opstr}{repr(self.expr)}>"
        else:
            return f"{name}<{repr(self.expr)}{self.opstr}>"

    def __str__(self) -> str:
        if self.fixity == Fixity.PREFIX:
            return f"{self.opstr}{self.expr}"
        else:
            return f"{self.expr}{self.opstr}"

    @property
    def value(self):
        return self.operation(self.expr)

    def clone(self) -> "UnaryExpr":
        return UnaryExpr(self.opstr, self.fixity, self.expr.clone())

    @staticmethod
    def get_operation(opstr) -> typing.Callable:
        return {
            "-": lambda expr: operator.neg(expr),
            "a": lambda roll: max(roll.rolls),
            "d": lambda roll: min(roll.rolls),
            "s": lambda roll: sum(roll.rolls),
        }[opstr]


class RollUnaryExpr(UnaryExpr):
    def __init__(self, opstr: str, expr: Expr) -> None:
        super().__init__(opstr, Fixity.POSTFIX, expr)

        assert isinstance(expr, RollExpr)
        self.roll = expr


class SortExpr(RollUnaryExpr):
    OPERATOR = "s"

    def __init__(self, expr: Expr) -> None:
        super().__init__(SortExpr.OPERATOR, expr)

    def roll_str(self) -> str:
        return ", ".join(map(str, sorted(self.roll.rolls)))


class AdvDisExpr(RollUnaryExpr):
    def __init__(self, opstr: str, expr: Expr) -> None:
        super().__init__(opstr, expr)

        # AdvDisExprs are adv and disadv and both implicity raise the number
        # of die to at least 2.
        if self.roll.qty < 2:
            self.roll.qty = 2


class AdvExpr(RollUnaryExpr):
    OPERATOR = "a"

    def __init__(self, expr: Expr) -> None:
        super().__init__(AdvExpr.OPERATOR, expr)

    def expr_str(self):
        return "max" + self.roll.expr_str(False)


class DisadvExpr(RollUnaryExpr):
    OPERATOR = "d"

    def __init__(self, expr: Expr) -> None:
        super().__init__(DisadvExpr.OPERATOR, expr)

    def expr_str(self):
        return "min" + self.roll.expr_str(False)


# List of OpExprSpecifications to describe different operators.
OPERATOR_SPECIFICATIONS: typing.List[OpExprSpecification] = []

# Current precedence:
# 1: k, a, d
# 2: ^
# 3: *, /
# 4: +, -
# 5: - (unary)
OPERATOR_SPECIFICATIONS.extend(
    [
        OpExprSpecification(
            o, p, Fixity.INFIX, [Expr, Expr], functools.partial(OpExpr, o)
        )
        for (o, p) in [("^", 2), ("*", 3), ("/", 3), ("+", 4), ("-", 4)]
    ]
)
OPERATOR_SPECIFICATIONS.append(
    OpExprSpecification(
        KeepExpr.OPERATOR, 1, Fixity.INFIX, [RollExpr, Expr], KeepExpr
    )
)
OPERATOR_SPECIFICATIONS.append(
    OpExprSpecification(
        "-",
        5,
        Fixity.PREFIX,
        [Expr],
        functools.partial(UnaryExpr, "-", Fixity.PREFIX),
    )
)
OPERATOR_SPECIFICATIONS.extend(
    [
        OpExprSpecification(
            o,
            1,
            Fixity.POSTFIX,
            [RollExpr],
            c,
        )
        for (o, c) in [
            (AdvExpr.OPERATOR, AdvExpr),
            (DisadvExpr.OPERATOR, DisadvExpr),
            (SortExpr.OPERATOR, SortExpr),
        ]
    ]
)


def consume(c: str, current_token: typing.Optional[Token]) -> TokenPair:
    """Consume a character, returning a finished and current token."""

    if current_token:
        return current_token.consume(c)
    else:
        return None, Token.from_char(c)


def tokenize(string: str):
    """Parse a string into valid tokens."""

    tokens: typing.List[Token] = []
    current_token: typing.Optional[Token] = None

    for c in string:
        finished_token, current_token = consume(c, current_token)
        if finished_token:
            tokens.append(finished_token)

    if current_token:
        finished_token, _ = current_token.consume(" ")

        if finished_token:
            tokens.append(finished_token)

    return tokens


def either_side(i, l):
    """Returns a None padded tuple of the three elements around index i."""

    if i == 0:
        a = None
    else:
        a = l[i - 1]

    if i == len(l) - 1:
        c = None
    else:
        c = l[i + 1]

    return a, l[i], c


def filter_nones(l):
    """Remove None elements of list l."""

    return [e for e in l if e is not None]


def replace_around(i, l, r, f_nones=True):
    """Replace the three elements centered at i with r."""

    if f_nones:
        r = filter_nones(r)

    return l[: max(i - 1, 0)] + r + l[i + 2 :]


def parse_non_operators(
    tokens: typing.List[Token],
) -> typing.List[typing.Union[Expr, OperatorToken]]:
    """Parse a list of Tokens into Exprs, leaving OperatorTokens untouched."""

    exprs: typing.List[typing.Union[Expr, OperatorToken]] = []
    depth = 0
    current_expr: typing.List[Token] = []
    for token in tokens:
        if isinstance(token, OpenExprToken):
            depth += 1
        elif isinstance(token, CloseExprToken):
            depth -= 1
            if depth == 0:
                exprs.append(SuperExpr(parse_tokens(current_expr)))
                current_expr = []
        elif depth:
            current_expr.append(token)
        elif isinstance(token, RollToken):
            exprs.append(RollExpr.from_token(token))
        elif isinstance(token, TerminalToken):
            exprs.append(ConstExpr(token.value()))
        elif isinstance(token, OperatorToken):
            exprs.append(token)
        else:
            raise ValueError(f"Can't handle token of type: {type(token)}")

    return exprs


def parse_operators(
    exprs: typing.List[typing.Union[Expr, OperatorToken]]
) -> typing.List[Expr]:
    """Attempt to associate OperatorTokens with their operands."""

    # Left operand, operator and right operand variables used while scanning
    # through expression list.
    a = b = c = None

    # Collapse operators with their operands, in order of precedence
    p = 1
    while p <= OpExprSpecification.MAX_PRECEDENCE:
        i = 0
        while i < len(exprs):
            a, b, c = either_side(i, exprs)
            for spec in OPERATOR_SPECIFICATIONS:
                if spec.precedence == p and spec.satisfied(a, b, c):
                    collapsed, index_delta = spec.instantiate(a, b, c)
                    exprs = replace_around(i, exprs, collapsed)
                    i += index_delta
                    break
            i += 1
        p += 1

    for expr in exprs:
        if not isinstance(expr, Expr):
            raise ValueError(f"Invalid operands for operator {expr.opstr}.")

    # mypy doesn't know that the union type is restricted by the above for loop,
    # so it flags this return
    return exprs  # type: ignore


# in an expression like 6 4d6k3, the const is 6 and indicates that the following
# expression should be repeated 6 times. MAX_CONST is the maximum value such a
# constant can take.
MAX_CONST = 10

# The const rule is that <const> <expr> becomes <const> instances of <expr> if
# expr contains any rolls.
def apply_const_rule(exprs: typing.List[Expr]):
    """Replace <const> <expr> with <const> instances of <expr>."""

    i = 0
    while i < len(exprs):
        a, b, c = either_side(i, exprs)

        if (
            isinstance(b, ConstExpr)
            and type(b.total) is int
            and c
            and c.has_rolls()
        ):
            n = b.total

            # ignore type of n through this block because mypy doesn't like the
            # assert here for some reason.

            assert type(n) is int

            if n > MAX_CONST:
                raise ValueError(
                    f"Can't repeat an expression more than {MAX_CONST} times."
                )

            # replace the constant and expr with constant.total of the expr
            exprs = replace_around(
                i, exprs, [a] + [c.clone() for _ in range(n)]  # type: ignore
            )

            # move the pointer along to the end of the new elements
            i += n - 1  # type: ignore
        i += 1

    return exprs


def parse_tokens(tokens: typing.List[Token]) -> typing.List[Expr]:
    """Parse a string of roll expressions."""

    return apply_const_rule(parse_operators(parse_non_operators(tokens)))


def parse(string: str) -> typing.List[Expr]:
    return parse_tokens(tokenize(string.lower()))


def get_rolls(string: str, max_qty: int = None) -> typing.List[Expr]:
    """Parse the input string, returning up to max_qty rolls."""

    return parse(string)[:max_qty]


def get_roll(string: str) -> Expr:
    """Parse for one roll or throw ValueError."""

    rolls = get_rolls(string, 1)
    if rolls:
        return rolls[0]
    raise ValueError


def calculate_total(exprs: typing.List[Expr]) -> Number:
    """Calculate to total result of a list of Expr objects."""

    return sum(expr.total for expr in exprs)


def get_result(string: str) -> Number:
    """Calculate and return the total of rolls parsed from a string."""

    return calculate_total(get_rolls(string))


def pad_to_longest(strings: typing.List[str]) -> typing.List[str]:
    """Space pad a list of strings to the same length."""

    length = max(len(s) for s in strings)
    return [s.ljust(length) for s in strings]


def rolls_string(rolls: typing.List[Expr], sep=SEP) -> str:
    """Return a descriptive string for a list of Roll objects."""

    if not rolls:
        return ""
    elif len(rolls) == 1:
        roll = rolls[0]
        return f"{roll}{sep}{roll.roll_str()}{sep}Total: {roll.total}"

    desc_strs = pad_to_longest([str(expr) for expr in rolls])
    expr_strs = pad_to_longest([expr.expr_str() for expr in rolls])
    roll_strs = pad_to_longest([expr.roll_str() for expr in rolls])

    string = ""
    for desc_str, expr_str, roll_str, roll in zip(
        desc_strs, expr_strs, roll_strs, rolls
    ):
        if len(roll.roll_info()) > 1:
            value_str = expr_str
        else:
            value_str = roll_str

        string += f"{desc_str}{sep}{value_str}{sep}Total: {roll.total}\n"

    string += f"Grand Total: {calculate_total(rolls)}"
    return string

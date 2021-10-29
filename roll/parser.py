import enum
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
        if c == " ":
            return None
        elif c.isnumeric():
            return Integer(c)
        elif c == ".":
            return Decimal(c)
        else:
            return NonTerminal.from_char(c)


class Terminal(Token):
    def value(self) -> typing.Any:
        raise NotImplementedError()


class Integer(Terminal):
    def __init__(self, token=""):
        super().__init__(token=token)
        assert not token or token.isnumeric()

    def consume(self, c: Char) -> TokenPair:
        if c.isnumeric():
            self.token += c
            return None, self
        elif c == ".":
            return None, Decimal(self.token + c)
        elif c == "d":
            return None, Roll(self.token + c)
        else:
            return self, Token.from_char(c)

    def value(self) -> int:
        return int(self.token)


class Decimal(Terminal):
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


class Roll(Terminal):
    SEPERATOR = "d"

    def __init__(self, token):
        super().__init__(token=token)
        qty, d, size = token.partition(Roll.SEPERATOR)
        assert not qty or qty.isnumeric()
        assert d == "d"
        assert not size or size.isnumeric()

    @property
    def qty(self):
        return int(self.token.partition(Roll.SEPERATOR)[0])

    @property
    def size(self):
        return int(self.token.partition(Roll.SEPERATOR)[2])

    def consume(self, c: Char) -> TokenPair:
        if c.isnumeric():
            self.token += c
            return None, self
        else:
            return self, Token.from_char(c)


class NonTerminal(Token):
    @staticmethod
    def from_char(c: Char) -> typing.Optional[Token]:
        if c == "(":
            return OpenExpr()
        elif c == ")":
            return CloseExpr()
        else:
            return Operator.from_char(c)


class OpenExpr(NonTerminal):
    def __init__(self):
        super().__init__(token="(")


class CloseExpr(NonTerminal):
    def __init__(self):
        super().__init__(token=")")


class Operator(NonTerminal):
    def __init__(self, token):
        super().__init__(token=token)

    @property
    def opstr(self):
        return self.token

    @staticmethod
    def from_char(c: Char) -> typing.Optional[Token]:
        if c in "+-*/k":
            return Operator(c)
        else:
            raise ValueError(f"Invalid character: {c}")


class Expr:
    def __repr__(self) -> str:
        return f"{type(self).__name__}<{str(self)}>"

    def __str__(self):
        return str(self.value)

    @property
    def value(self) -> Number:
        raise NotImplementedError()

    @property
    def result(self) -> Number:
        value = self.value
        if value // 1 == value:
            return int(value)
        return round(value, 2)

    def roll_info(self) -> typing.List[RollInfo]:
        return []


class TerminalExpr(Expr):
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

    @property
    def value(self) -> Number:
        return self._value


class RollExpr(TerminalExpr):
    MAX_QTY = 1000

    def __init__(self, qty: int, size: int):
        super().__init__()
        self.qty = qty
        self.size = size
        self._rolls: typing.Optional[typing.List[int]] = None

        if self.qty > RollExpr.MAX_QTY:
            raise ValueError("Maximum quantity exceeded.")

    def __eq__(self, o: object) -> bool:
        return (
            type(o) == type(self)
            and isinstance(o, RollExpr)
            and o.qty == self.qty
            and o.size == self.size
        )

    def __str__(self) -> str:
        return f"{self.qty}{Roll.SEPERATOR}{self.size}"

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

    def roll_info(self) -> typing.List[RollInfo]:
        return [(str(self), self.rolls)]

    def reroll(self, n: int) -> typing.Tuple[typing.List[int], Number]:
        # accessing self.rolls means that self._rolls is not None, though
        # mypy doesn't know this.

        for r in heapq.nsmallest(n, self.rolls):
            self._rolls[  # type: ignore
                self._rolls.index(r)  # type: ignore
            ] = random.randint(1, self.die)

        return self._rolls, self.calculate_total()  # type: ignore

    @staticmethod
    def from_token(token: Roll):
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
            + " + ".join([f"({repr(expr)})" for expr in self.exprs])
            + ">"
        )

    def __str__(self):
        return " + ".join([f"({expr})" for expr in self.exprs])

    @property
    def value(self) -> Number:
        return sum(expr.value for expr in self.exprs)


class Fixity(enum.Enum):
    INFIX = enum.auto()  # a op b
    PREFIX = enum.auto()  # op a
    POSTFIX = enum.auto()  # a op


class OpExprSpecification:
    MAX_PRECEDENCE = 0

    def __init__(
        self,
        opstr: Char,
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
        a: typing.Union[Expr, Operator, None],
        b: typing.Union[Expr, Operator, None],
        c: typing.Union[Expr, Operator, None],
    ) -> bool:
        if not isinstance(b, Operator) or b.opstr != self.opstr:
            return False

        if self.fixity is Fixity.INFIX:
            return isinstance(a, self.left) and isinstance(c, self.right)
        elif self.fixity is Fixity.PREFIX:
            return isinstance(c, self.arg)
        elif self.fixity is Fixity.POSTFIX:
            return isinstance(a, self.arg)

    def instantiate(
        self,
        a: typing.Union[Expr, Operator, None],
        b: typing.Union[Expr, Operator, None],
        c: typing.Union[Expr, Operator, None],
    ) -> typing.Tuple[typing.List[Expr], int]:
        # The typing ignores in this function are to suppress type issues which
        # are handled by the below assertion.

        assert self.satisfied(a, b, c)

        if self.fixity is Fixity.INFIX:
            return [self.klasse(b.opstr, a, c)], -1  # type: ignore
        elif self.fixity is Fixity.PREFIX:
            return [a, self.klasse(b.opstr, c)], 0  # type: ignore
        elif self.fixity is Fixity.POSTFIX:
            return [self.klasse(b.opstr, a), c], 0  # type: ignore


class OpExpr(NonTerminalExpr):
    def __init__(self, opstr: Char, left: Expr, right: Expr) -> None:
        super().__init__([left, right])
        self.opstr = opstr
        self.left = left
        self.right = right

        self.operation = OpExpr.get_operation(opstr)

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
        return self.operation(self.left.value, self.right.value)

    def roll_info(self) -> typing.List[RollInfo]:
        return self.left.roll_info() + self.right.roll_info()

    @staticmethod
    def get_operation(opstr) -> typing.Callable:
        return {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }[opstr]


class KeepExpr(NonTerminalExpr):
    OPERATOR = "k"

    def __init__(self, opstr, roll, keep) -> None:
        assert isinstance(roll, RollExpr)
        assert opstr == KeepExpr.OPERATOR

        super().__init__([roll, keep])

        self.roll = roll
        self.keep = keep

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}<{repr(self.roll)}{KeepExpr.OPERATOR}"
            f"{repr(self.keep)}>"
        )

    def __str__(self) -> str:
        return f"{self.roll}{KeepExpr.OPERATOR}{self.keep}"

    @property
    def value(self) -> Number:
        return sum(heapq.nlargest(int(self.keep.value), self.roll.rolls))


# List of OpExprSpecifications to describe different operators.
OPERATOR_SPECIFICATIONS: typing.List[OpExprSpecification] = []

OPERATOR_SPECIFICATIONS.extend(
    [
        OpExprSpecification(o, p, Fixity.INFIX, [Expr, Expr], OpExpr)
        for (o, p) in [("*", 2), ("/", 2), ("+", 3), ("-", 3)]
    ]
)
OPERATOR_SPECIFICATIONS.append(
    OpExprSpecification(
        KeepExpr.OPERATOR, 1, Fixity.INFIX, [RollExpr, Expr], KeepExpr
    )
)


def consume(c: str, current_token: typing.Optional[Token]) -> TokenPair:
    if current_token:
        return current_token.consume(c)
    else:
        return None, Token.from_char(c)


def tokenize(string: str):
    tokens: typing.List[Token] = []
    current_token: typing.Optional[Token] = None

    for c in string:
        finished_token, current_token = consume(c, current_token)
        if finished_token:
            tokens.append(finished_token)

    if current_token:
        tokens.append(current_token)

    return tokens


def either_side(i, l):
    if i == 0:
        a = None
    else:
        a = l[i - 1]

    if i == len(l) - 1:
        c = None
    else:
        c = l[i + 1]

    return a, l[i], c


def parse_tokens(tokens: typing.List[Token]):
    exprs: typing.List[typing.Union[Expr, Operator]] = []
    depth = 0
    current_expr: typing.List[Token] = []
    for token in tokens:
        if isinstance(token, OpenExpr):
            depth += 1
        elif isinstance(token, CloseExpr):
            depth -= 1
            if depth == 0:
                exprs.append(SuperExpr(parse_tokens(current_expr)))
                current_expr = []
        elif depth:
            current_expr.append(token)
        elif isinstance(token, Roll):
            exprs.append(RollExpr.from_token(token))
        elif isinstance(token, Terminal):
            exprs.append(ConstExpr(token.value()))
        elif isinstance(token, Operator):
            exprs.append(token)
        else:
            raise ValueError(f"Can't handle token of type: {type(token)}")

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
                    exprs = [
                        *exprs[: i - 1],
                        *collapsed,
                        *exprs[i + 2 :],
                    ]
                    i += index_delta
                    break
            i += 1
        p += 1

    # remove the Nones added as padding
    return exprs


def parse(string: str):
    return parse_tokens(tokenize(string))


print(parse("2d12 k3+4*(3 / 2) 2d4")[0])

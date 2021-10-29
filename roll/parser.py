import heapq
import operator
import random
import typing

# Generic numeric base type
Number = typing.Union[int, float]

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

    def consume(self, c: str) -> TokenPair:
        return self, Token.from_char(c)

    @staticmethod
    def from_char(c: str) -> typing.Optional["Token"]:
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

    def consume(self, c: str) -> TokenPair:
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

    def consume(self, c: str) -> TokenPair:
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

    def consume(self, c: str) -> TokenPair:
        if c.isnumeric():
            self.token += c
            return None, self
        else:
            return self, Token.from_char(c)


class NonTerminal(Token):
    @staticmethod
    def from_char(c: str) -> typing.Optional[Token]:
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
    PRECEDENCE = {"k": 1, "*": 2, "/": 2, "+": 3, "-": 3}
    MAX_PRECEDENCE = 3

    def __init__(self, token):
        super().__init__(token=token)
        self.precedence = Operator.PRECEDENCE[token]

    @staticmethod
    def from_char(c: str) -> typing.Optional[Token]:
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


class OpExpr(NonTerminalExpr):
    def __init__(self, opstr: str, left: Expr, right: Expr) -> None:
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
    def from_token(left: Expr, operator: Operator, right: Expr) -> Expr:
        if operator.token == KeepExpr.CHAR:
            return KeepExpr(left, right)
        else:
            return OpExpr(operator.token, left, right)

    @staticmethod
    def get_operation(opstr) -> typing.Callable:
        return {
            "+": operator.add,
            "-": operator.sub,
            "*": operator.mul,
            "/": operator.truediv,
        }[opstr]


class KeepExpr(NonTerminalExpr):
    CHAR = "k"

    def __init__(self, roll, keep) -> None:
        assert isinstance(roll, RollExpr)

        super().__init__([roll, keep])

        self.roll = roll
        self.keep = keep

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}<{repr(self.roll)}{KeepExpr.CHAR}"
            f"{repr(self.keep)}>"
        )

    def __str__(self) -> str:
        return f"{self.roll}{KeepExpr.CHAR}{self.keep}"

    @property
    def value(self) -> Number:
        return sum(heapq.nlargest(int(self.keep.value), self.roll.rolls))


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

    # Collapse operators with their operands, in order of precedence
    p = 1
    while p <= Operator.MAX_PRECEDENCE:
        i = 1
        while i < len(exprs) - 1:
            left, op, right = exprs[i - 1 : i + 2]
            if isinstance(op, Operator) and op.precedence == p:
                try:
                    assert isinstance(left, Expr)
                    assert isinstance(right, Expr)

                    exprs = [
                        *exprs[: i - 1],
                        OpExpr.from_token(left, op, right),
                        *exprs[i + 2 :],
                    ]
                except AssertionError:
                    raise ValueError(
                        f"Invalid operands for {op.token}: {left} and {right}."
                    )

                i -= 1
            i += 1
        p += 1

    return exprs


def parse(string: str):
    return parse_tokens(tokenize(string))


print(parse("2d12 k3+4*(3 / 2) 2d4")[1].roll_info())

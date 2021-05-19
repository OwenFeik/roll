import heapq
import operator
import random
import re
import typing

numeric = typing.Union[int, float]


class Modifier:
    """Stores a value and operation and can apply itself to another value."""

    def __init__(self, val: numeric, opstr: str):
        self.val = val
        self.opstr = opstr
        self.operation = get_operator(opstr)

    def __str__(self):
        return f"{self.opstr} {self.val}"

    def apply(self, val: numeric) -> numeric:
        return self.operation(val, self.val)

    @staticmethod
    def from_string(string: str) -> "Modifier":
        return Modifier(int(string[1:]), string[0])


class Roll:
    """Stores information and returns results for a parsed roll."""

    MAX_QTY = 1000

    def __init__(
        self,
        qty: int,
        die: int,
        adv: bool,
        disadv: bool,
        keep: int,
        modifiers: typing.List[Modifier],
        operation: typing.Callable,
    ):
        self.qty = qty
        self.die = die
        self.adv = adv
        self.disadv = disadv
        self.keep = keep
        self.modifiers = modifiers
        self.operation = operation

        if self.qty > Roll.MAX_QTY:
            raise ValueError("Maximum quantity exceeded.")

        self.resolved = False
        self._rolls: typing.List[int] = []
        self._total: numeric = 0

    def __str__(self):
        return self.full_str()

    def __repr__(self):
        return f"<{str(self)}>"

    def full_str(self, pad_desc: int = 0, pad_roll: int = 0) -> str:
        string = self.desc_str(pad_desc) + "\t" + self.roll_str(pad_roll)

        if self.modifiers or len(self.rolls) > 1:
            string += f"\tTotal: {clean_number(self.total)}"

        return string

    def desc_str(self, pad_to: int = 0) -> str:
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
        for mod in self.modifiers:
            string += " " + str(mod)

        while len(string) < pad_to:
            string += " "

        return string

    def dice_str(self) -> str:
        # Used for saving results in database
        return f"{self.qty}d{self.die}"

    def roll_str(self, pad_to: int = 0) -> str:
        if len(self.rolls) > 1:
            string = f"Rolls: {str(self.rolls)[1:-1]}"
        else:
            string = f"Roll: {self.rolls[0]}"

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

    def ensure_rolled(self) -> None:
        if not self.resolved:
            self._rolls = [random.randint(1, self.die) for _ in range(self.qty)]
            self.calculate_total()
            self.resolved = True

    def resolve(self) -> typing.Tuple[typing.List[int], numeric]:
        self.ensure_rolled()
        return self._rolls, self._total

    def reroll(self, n: int) -> typing.Tuple[typing.List[int], numeric]:
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


def get_operator(opstr: str) -> typing.Callable:
    """Return a function related to a given numeric operation string."""

    # For the inverse division operator (hack), division is simply called
    # with the operands reversed. Pylint assumes that this is a mistake when
    # it is in fact intended.
    #
    # pylint: disable=arguments-out-of-order

    return {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "\\": lambda a, b: operator.truediv(b, a),
    }[opstr]


def clean_number(num: numeric) -> numeric:
    """Round to 2 or less decimal places."""

    if num // 1 == num:
        return int(num)
    return round(num, 2)


def parse_mods(modstr: str) -> typing.Tuple[typing.List[Modifier], int]:
    """Parse a string of modifier statements."""

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

    return mods, keep


def get_rolls(string: str, max_qty: int = -1) -> typing.List[Roll]:
    """Parse a string for roll statements, returning a list of Roll objects."""

    rolls = []

    # Explanation of groups in the below regex:
    #
    # const: a value before the roll followed by an operator. only applies for
    #   single rolls. e.g. <2> + 2d20
    # op: the operation to apply to the previous term in the string and the
    #   roll. e.g. 2 <+> 2d20 <*> d10
    # n: the number of times to do this roll. must be followed by a space.
    #   e.g. <2> d20 <6> 4d6k3
    # qty: the number of dice to roll
    #   e.g. <2>d10 <8>d6
    # die: the size of dice to roll
    #   e.g. 2d<10> 8d<6>
    # advstr: annotation to indicate whether the roll is at advantage or
    #   disadvantage. any number of as and ds. e.g. d20<a> d20<addd>
    # mods: modifiers of the form operator, value or keep x.
    #   e.g. d20 <+ 8 - 2> 10d6<k8 + 2>
    #
    # The expression accepts any number of spaces between terms.
    regex = re.compile(
        r"(?P<const>\d+(?= *[+-/*]))? *(?P<op>[+-/*]?)? *(?P<n>\d+"
        r"(?= +))? *(?P<qty>\d*)d(?P<die>\d+) *(?P<advstr>[ad]*)"
        r"(?P<mods>( *(k *\d+|[+-/*] *\d+(?!d)))*(?![\dd]))",
        flags=re.IGNORECASE,
    )

    empty_group = lambda g: g in [None, ""]

    count = 0
    for roll in re.finditer(regex, string.lower()):

        n = 1 if empty_group(roll.group("n")) else int(roll.group("n"))
        qty = 1 if empty_group(roll.group("qty")) else int(roll.group("qty"))

        if qty == 0:
            raise ValueError("I can't roll a zero sided die.")

        die = int(roll.group("die"))

        advstr = roll.group("advstr")
        adv = "a" in advstr
        disadv = "d" in advstr

        # if under advantage and disadvantage, roll is straight
        if adv and disadv:
            adv = disadv = False

        # a roll like d20a implicitly means 2d20a
        if (adv or disadv) and qty == 1:
            qty = 2

        mods, keep = parse_mods(roll.group("mods"))

        opstr = "+" if empty_group(roll.group("op")) else roll.group("op")
        op = get_operator(opstr)

        const = roll.group("const")
        if not empty_group(const):
            if opstr == "/":
                opstr = "\\"

            special_mods = mods[:]

            special_mods.insert(0, Modifier(int(const), opstr))
            op = get_operator("+")

            n -= 1
            rolls.append(Roll(qty, die, adv, disadv, keep, special_mods, op))

        for _ in range(n):
            rolls.append(Roll(qty, die, adv, disadv, keep, mods[:], op))

        count += 1
        if max_qty > 0 and count == max_qty:
            break

    return rolls


def calculate_total(rolls: typing.List[Roll]) -> numeric:
    """Calculate to total result of a list of Roll objects."""

    total = 0
    for r in rolls:
        total = r.apply(total)

    return total

def get_result(string: str) -> numeric:
    """Calculate and return the total of rolls parsed from a string."""

    return calculate_total(get_rolls(string))


def rolls_string(rolls: typing.List[Roll]) -> str:
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

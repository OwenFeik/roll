import unittest

from roll import *


class TestParser(unittest.TestCase):
    def test_tokenize_arithmetic(self) -> None:
        self.assertEqual(
            tokenize("1 / 2.72 - (25 + 38) * 3.14"),
            [
                IntegerToken("1"),
                OperatorToken("/"),
                DecimalToken("2.72"),
                OperatorToken("-"),
                OpenExprToken(),
                IntegerToken("25"),
                OperatorToken("+"),
                IntegerToken("38"),
                CloseExprToken(),
                OperatorToken("*"),
                DecimalToken("3.14"),
            ],
        )

    def test_tokenize_roll(self) -> None:
        self.assertEqual(
            tokenize("2 2d12 * 3 + 2"),
            [
                IntegerToken("2"),
                RollToken("2d12"),
                OperatorToken("*"),
                IntegerToken("3"),
                OperatorToken("+"),
                IntegerToken("2"),
            ],
        )

    def test_parse(self) -> None:
        self.assertEqual(
            parse("-2d12 k3+4*(3 / 2) 2d4"),
            [
                UnaryExpr(
                    "-",
                    Fixity.PREFIX,
                    OpExpr(
                        "+",
                        KeepExpr(RollExpr(2, 12), ConstExpr(3)),
                        OpExpr(
                            "*",
                            ConstExpr(4),
                            SuperExpr(
                                [OpExpr("/", ConstExpr(3), ConstExpr(2))]
                            ),
                        ),
                    ),
                ),
                RollExpr(2, 4),
            ],
        )

    def test_evaluation(self) -> None:
        self.assertEqual(
            parse("10d1 + 4 ^ 2 * 3 / 2")[0].value, 10 + 4 ** 2 * 3 / 2
        )

    def test_const_rule(self) -> None:
        self.assertEqual(parse("2 d4"), [RollExpr(1, 4)] * 2)

    def test_advantage(self) -> None:
        exprs = parse("d20a")
        self.assertEqual(len(exprs), 1)
        expr = exprs[0]
        self.assertEqual(expr.value, max(expr.roll_info()[0][1]))

    def test_disadvantage(self) -> None:
        exprs = parse("d20d")
        self.assertEqual(len(exprs), 1)
        expr = exprs[0]
        self.assertEqual(expr.value, min(expr.roll_info()[0][1]))

    def test_missing_operand(self) -> None:
        self.assertRaises(ValueError, parse, "+ 2")

    def test_operator_operand(self) -> None:
        self.assertRaises(ValueError, parse, "- + 2")

    def test_too_many_die(self) -> None:
        self.assertRaises(ValueError, parse, "1001d20")

    def test_rolls_string(self) -> None:
        self.assertEqual(
            rolls_string(get_rolls("4d1")), "4d1\tRolls: 1, 1, 1, 1\tTotal: 4"
        )

    def test_multiple_rolls_string(self) -> None:
        self.assertEqual(
            rolls_string(get_rolls("2 2d1")),
            "2d1\tRolls: 1, 1\tTotal: 2\n"
            "2d1\tRolls: 1, 1\tTotal: 2\n"
            "Grand Total: 4",
        )


if __name__ == "__main__":
    unittest.main()

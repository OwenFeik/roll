import unittest

from roll.parser import *


class TestParser(unittest.TestCase):
    def test_tokenize_arithmetic(self) -> None:
        self.assertEqual(
            tokenize("1 / 2.72 - (25 + 38) * 3.14"),
            [
                Integer("1"),
                Operator("/"),
                Decimal("2.72"),
                Operator("-"),
                OpenExpr(),
                Integer("25"),
                Operator("+"),
                Integer("38"),
                CloseExpr(),
                Operator("*"),
                Decimal("3.14"),
            ],
        )

    def test_tokenize_roll(self) -> None:
        self.assertEqual(
            tokenize("2 2d12 * 3 + 2"),
            [
                Integer("2"),
                Roll("2d12"),
                Operator("*"),
                Integer("3"),
                Operator("+"),
                Integer("2"),
            ],
        )

    def test_parse(self) -> None:
        self.assertEqual(
            parse("-2d12 k3+4*(3 / 2) 2d4"),
            [
                UnaryNegateExpr(
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
                    )
                ),
                RollExpr(2, 4),
            ],
        )

    def test_evaluation(self) -> None:
        self.assertEqual(
            parse("10d1 + 4 ^ 2 * 3 / 2")[0].value, 10 + 4 ** 2 * 3 / 2
        )

    def test_invalid(self) -> None:
        self.assertRaises(ValueError, parse, "- + 2")

    def test_too_many_die(self) -> None:
        self.assertRaises(ValueError, parse, "1001d20")


if __name__ == "__main__":
    unittest.main()

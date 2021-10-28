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
                Decimal("3.14")
            ]
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
                Integer("2")
            ]
        )

if __name__ == "__main__":
    unittest.main()

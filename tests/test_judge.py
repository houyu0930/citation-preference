"""Regression checks for verdict parsing; no model server is required."""

import unittest

from common.judge import parse_choice


class ParseChoiceTests(unittest.TestCase):
    def test_explicit_verdicts(self):
        for output, expected in [
            ("A", "A"),
            ("B.", "B"),
            ('"B"', "B"),
            ("(a)", "A"),
            ("**B**", "B"),
            ("Answer: B", "B"),
            ("Answer: **B**", "B"),
            ("Answer: Output (b)", "B"),
            ("B\n\n**Rationale:**\nBoth outputs are relevant.", "B"),
        ]:
            with self.subTest(output=output):
                self.assertEqual(parse_choice(output), expected)

    def test_prose_is_not_a_verdict(self):
        for output in [None, "", " \n ", "Both responses are comparable.",
                       "A is weaker than B.", "Answer: neither", "A or B"]:
            with self.subTest(output=output):
                self.assertIsNone(parse_choice(output))

    def test_cot_uses_final_answer(self):
        self.assertEqual(
            parse_choice("Answer: A\nAfter comparison:\nAnswer: **B**", "cot"),
            "B",
        )
        self.assertEqual(parse_choice("A is weaker than B.\n\nB", "cot"), "B")
        self.assertIsNone(parse_choice("Both responses are comparable.", "cot"))


if __name__ == "__main__":
    unittest.main()

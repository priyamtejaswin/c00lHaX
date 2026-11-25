# --- Day 1: Not Quite Lisp ---

import sys
from pathlib import Path


input_path = Path(sys.argv[1])
floor = 0
first_basement_ix = 0
found_basement = False

with open(input_path, 'r', encoding="utf-8") as fp:
    for line in fp:
        for char in line:
            if char == "(":
                floor += 1
            elif char == ")":
                floor -= 1
            else:
                raise ValueError(f"Expected '(' or ')', got {char}")

            if found_basement is False:
                first_basement_ix += 1

                if floor < 0:
                    found_basement = True

print(f"Final floor tally {floor}")
print(f"Basement character position {first_basement_ix}")

#!/usr/bin/env python3
import sys

eqns = []

catch = False
for line in sys.stdin:
    if line.startswith("$$"):
        eqns.append(line)
        if catch is False:
            catch = True
        else:
            catch = False
            if len(eqns) >= 4 and "begin{align*}" in eqns[1] and "end{align*}" in eqns[-2]:
                for e in eqns[1:-1]:
                    sys.stdout.write(e)
            else:
                for e in eqns:
                    sys.stdout.write(e)

            eqns = []
    else:
        if catch is False:
            sys.stdout.write(line)
        else:
            eqns.append(line)
sys.stdout.write("\n")

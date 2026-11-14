#!/bin/bash
cd "$(dirname "$0")"
until [ -f out/sky2_san_francisco_c03.done ]; do sleep 30; done
for c in munich etoile florence san_francisco; do OMP_NUM_THREADS=2 python3 pol_paths.py $c --users 60 --dirs 24 --tag _c03 --wall concrete --thick 0.3 --src _c03 2>&1 | grep -E "Traceback|Error|$c _c03 60"; done
echo POLDONE

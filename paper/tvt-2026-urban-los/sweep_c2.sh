#!/bin/bash
# Facade sweep on Munich, 60 users (the same users as the main run), 1000 m plane, 1.13 deg cells, 1e9 rays.
cd "$(dirname "$0")"
B="--planeOff 1000 --cell 2.5 --samples 1e9"
r(){ t=$1; shift; [ -f out/sky2_munich$t.npz ] && [ "$FORCE" != 1 ] || OMP_NUM_THREADS=2 python3 sky_maps2.py munich 60 $B --tag $t "$@" 2>&1 | grep -E "done|Error|error|Traceback" ; }
r _wm005 --thick 0.05
r _wm015 --thick 0.15
r _wm02  --thick 0.2
r _wm03  --thick 0.3
r _wm05  --thick 0.5
r _wmhs  --thick 5
r _wc01  --wall concrete --thick 0.1
r _wc02  --wall concrete --thick 0.2
r _wc03  --wall concrete --thick 0.3
r _wchs  --wall concrete --thick 5
r _wb03  --wall brick --thick 0.3
r _wg03  --wall glass --thick 0.01
r _wgnd  --groundThick 5
r _wc03r --wall concrete --thick 0.3 --roof concrete
r _wc03g --wall concrete --thick 0.3 --groundThick 5
r _wc03s --wall concrete --thick 0.3 --scatter 0.4
r _wchss --wall concrete --thick 5 --scatter 0.4
r _wmhss --thick 5 --scatter 0.4
for f in 1.6 2.5 3.0 3.5; do r _wc03f$f --wall concrete --thick 0.3 --freq ${f}e9; r _wchsf$f --wall concrete --thick 5 --freq ${f}e9; done
for f in 2.5 3.0; do r _wm01f$f --freq ${f}e9; done
r _wc03d6 --wall concrete --thick 0.3 --depth 6
r _wchsd6 --wall concrete --thick 5 --depth 6
r _sd9 --depth 9
r _wchsd9 --wall concrete --thick 5 --depth 9
echo ALLDONE

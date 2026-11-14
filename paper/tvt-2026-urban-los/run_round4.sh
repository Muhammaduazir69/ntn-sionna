#!/bin/bash
# Round 4: concrete 0.3 m facades, smooth and rough, and a second user realization, 4 cities x 300 users.
cd "$(dirname "$0")"
B="--planeOff 1000 --cell 2.5 --samples 1e9"
for s in "_c03|--wall concrete --thick 0.3" "_c03s|--wall concrete --thick 0.3 --scatter 0.4" "_c03sd2|--wall concrete --thick 0.3 --seed 2"; do
  t=${s%%|*}; a=${s#*|}
  for c in munich etoile florence san_francisco; do
    [ -f out/sky2_$c$t.done ] && continue
    OMP_NUM_THREADS=2 python3 sky_maps2.py $c 300 $B --tag $t $a 2>&1 | grep -E "done|Traceback|Error" && touch out/sky2_$c$t.done
  done
done
for c in munich etoile florence san_francisco; do OMP_NUM_THREADS=2 python3 -c "import street_geom as g; g.run('$c','sky2','street2','_c03sd2')" 2>&1 | grep -v -i warn; done
echo R4DONE

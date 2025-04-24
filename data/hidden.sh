#!/bin/bash

for hidden in {1..35}
do
    python extract_esm.py --HIDDEN=$hidden
done
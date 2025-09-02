# ESM fine tuning
- targets: *Togaviridae* and *Phenuiviridae* (pan-family)
- Data format? FASTA? TSV?

## How to download protein sequences
Go [UniProt](https://www.uniprot.org) and query (taxonomy_id:?????), including parentheses.

### Togaviridae: 11018
- Alphavirus: 11019
- Unclassified: 2788734

### Phenuiviridae: 1980418
- Bandavirus: 2733256 (2461)
- Beidivirus: 2501981 (3)
- Bocivirus: 3153066 --- no results?
- Citricivirus: 3044688 (3)
- Coguvirus: 2560118 (96)
- Entovirus: 2733257 (3)
- Goukovirus: 1980420 (52)
- Horwuvirus: 2501976 (27)
- Ixovirus: 2733258 (18)
- Laulavirus: 2560166 (13)
- Lentinuvirus: 2733259 (10)
- Mechlorovirus: 3044759 (28)
- Mobuvirus: 2501979 (15)
- Phasivirus: 1980421 (160)
- Phlebovirus: 11584 (2856)
- Pidchovirus: 2501975 (3)
- Rubodvirus: 2733260 (84)
- Tanzavirus: 2843070 (3)
- Tenuivirus: 12329 (905)
- Uukuvirus: 2734594 (170)
- Wenrivirus: 2560254 (10)
- Unclassified: 2024603 (178)

## Extract ESM2 embeddings
run `scripts/extract.py`. But make sure that there is no "/" in the fasta annotation.
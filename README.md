# ESM fine tuning
- targets: *Togaviridae* and *Phenuiviridae* (pan-family)
- Data format? FASTA? TSV?

## How to download protein sequences
Go [UniProt](https://www.uniprot.org) and query (taxonomy_id:?????), including parentheses.

### Togaviridae: 11018
- Alphavirus: 11019
- Unclassified: 2788734

### Phenuiviridae: 1980418
- Bandavirus: 2733256
- Beidivirus: 2501981
- Bocivirus: 3153066
- Citricivirus: 3044688
- Coguvirus: 2560118
- Entovirus: 2733257
- Goukovirus: 1980420
- Horwuvirus: 2501976
- Ixovirus: 2733258
- Laulavirus: 2560166
- Lentinuvirus: 2733259
- Mechlorovirus: 3044759
- Mobuvirus: 2501979
- Phasivirus: 1980421
- Phlebovirus: 11584
- Pidchovirus: 2501975
- Rubodvirus: 2733260
- Tanzavirus: 2843070
- Tenuivirus: 12329
- Uukuvirus: 2734594
- Wenrivirus: 2560254
- Unclassified: 2024603

## Extract ESM2 embeddings
run `scripts/extract.py`. But make sure that there is no "/" in the fasta annotation.
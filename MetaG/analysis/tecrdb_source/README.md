`TECRDB.csv` — the NIST TECRDB extract (4,544 measurements) the benchmark was matched from; restored verbatim
from git (`6f7f94e:pipeline/TECRDB.csv`, the file `pipeline/build_tecrdb_set.py` read). Each row is one
measurement of K′ (or K) at its own T, pH, ionic strength and pMg. `../build_tecrdb_standard.py` transforms
every measurement to MetaG's conditions (pH 7, I = 0, no Mg²⁺) before aggregating.

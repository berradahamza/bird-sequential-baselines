# Vérifications

`python -m unittest discover -s tests -v` : contrôle cinq partitions, absence de
recouvrement entre visiteurs, tailles des jeux 2022..2026 et batches 80/78/75/78/80,
préfixes strictement causaux, deux architectures sans CL4SRec, stabilité padding,
contrôle de gradients, rangs/ties déterministes.

Le test Markov vérifie à 12 décimales que les trois méthodes retrouvent exactement
les Acc@1 et MRR de validation et test de
`STKG_CLL_POI/runs/baselines_seed2022/baselines.json`, disponible dans l'archive
source fournie au moment du développement. Aucun résultat neuronal annoncé tant
que les commandes de la campagne n'ont pas été exécutées sur le Dell.

Les SHA256 des données préparées et fichiers de split sont conservés dans les
configurations des runs et vérifiés lors de l'évaluation et de la reprise.

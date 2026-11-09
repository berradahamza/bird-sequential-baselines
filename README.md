# BIRD — baselines alignées sur la campagne STKG_CLL_POI

Ce paquet autonome contient **GRU4Rec-CE-prefix**, **SASRec-CE-prefix**, la popularité,
**Markov d'ordre 1** et **Markov d'ordre 2**. Il reprend le code de `BIRD_Baselines_Dell`
mais exclut entièrement CL4SRec. Les entrées JSON BIRD préparées sont incluses : pas de
retraitement, de modèle externe ou de clé API. Ces adaptations préfixe + CE ne sont pas des
reproductions exactes des logiciels originaux GRU4Rec/SASRec.

## Protocole strictement partagé avec STKG_CLL_POI

- **5 partitions indépendantes au niveau visiteur** : `split_seed = 2022, 2023, 2024, 2025, 2026`.
- **3 initialisations par partition et par réseau** : `seed = 2022, 2023, 2024`.
- **Batch size 64 ; maximum 30 époques ; patience 5** (amélioration stricte du **MRR validation**).
- Historique de 32 POI, embedding 64, dropout 0,2, learning rate 0,001, AdamW,
  weight decay 0,00005 ; catalogue complet 359 POI ; pas de masque de disponibilité.
- 35 visiteurs train, 8 validation, 8 test à chaque split ; aucun visiteur n'est partagé.
- Les partitions sont figées dans `splits/split_*.json`, régénérées par l'algorithme
  `STKG_CLL_POI/stkg_cll_poi/features.py::visitor_split` ; comparaison effectuée
  avec les fichiers de splits enregistrés dans l'archive CLL et avec la partition 2022 originale.
- Markov et la popularité : *comptages des transitions du train uniquement*, lissage
  par backoff `0.001` identique à STKG_CLL_POI ; calcul **une fois par split**
  pour validation et test (aucune seed d'initialisation).
- Métriques : Acc@1, Hit@5, Hit@10, MRR, agrégations micro, macro par visiteur,
  intervalles bootstrap groupés par visiteur. Les `test_metrics.json` neuronaux ne sont
  produits **qu'après entraînement**, avec checkpoint choisi en validation.

| Split | Préfixes train | Batches/époque (64) | Validation | Test |
|---|---:|---:|---:|---:|
| 2022 | 5 082 | 80 | 988 | 1 083 |
| 2023 | 4 955 | 78 | 1 122 | 1 076 |
| 2024 | 4 766 | 75 | 1 168 | 1 219 |
| 2025 | 4 942 | 78 | 1 075 | 1 136 |
| 2026 | 5 098 | 80 | 1 098 | 957 |

## Commandes Windows PowerShell (Dell)

Extraire le ZIP. Python 3.12 conseillé. Dans le répertoire `BIRD_Sequential_Baselines` :

```powershell
.\setup_windows.bat
.\.venv\Scripts\python.exe run_campaign.py plan
.\.venv\Scripts\python.exe run_campaign.py train --device auto
.\.venv\Scripts\python.exe run_campaign.py test --device auto
.\.venv\Scripts\python.exe run_campaign.py summary
```

Le mode `train` lance **30 entraînements** (2 modèles × 5 splits × 3 seeds) et les
**5 estimations statistiques** (3 méthodes, une seule fois par split). Le mode `test`
charge les meilleurs checkpoints sans mise à jour des poids. Les interruptions après
une époque complète peuvent être reprises en relançant exactement la commande `train`.
On peut aussi double-cliquer `train_all_windows.bat` puis `test_all_windows.bat`.

Avec NVIDIA CUDA, exécuter facultativement `setup_nvidia_windows.bat` (voir
l'environnement/pilote requis), puis `run_campaign.py train --device cuda`.
Sans NVIDIA, la variante CPU fonctionne aussi (plus lentement).

## Linux

```bash
bash setup_linux.sh
.venv/bin/python run_campaign.py plan
.venv/bin/python run_campaign.py train
.venv/bin/python run_campaign.py test
.venv/bin/python run_campaign.py summary
```

## Sorties

```text
runs/
  split2022/ ... split2026/
    baselines.json                # popularity, Markov1/2, validation ET test
    seed2022/ ... seed2024/
      gru4rec/
        config.json metrics.json best.pt last.pt history.csv training.log
        validation_predictions.csv test_metrics.json test_predictions.csv
      sasrec/ ...
  summary/
    results_per_run.csv           # une ligne / split / seed / modèle / fold
    results_per_split.csv         # moyenne sur les 3 seeds pour chaque split
    results_final.csv             # moyenne / écart-type entre les 5 splits
```

**Attention à l'agrégation** : les modèles neuronaux sont moyennés sur les 3 seeds
*à l'intérieur* de chaque split, puis les cinq moyennes sont agrégées.
Pour Popularité / Markov, il n'y a qu'une mesure par split : **pas de triplication**.
Les scripts marquent les synthèses incomplètes `complete=False`; ne pas utiliser ces
lignes comme résultat final. Les splits se recouvrent : leur écart-type est descriptif.

## Commandes ciblées (diagnostic)

```powershell
.\.venv\Scripts\python.exe run.py doctor --split-seed 2024
.\.venv\Scripts\python.exe run.py train --model sasrec --split-seed 2024 --seed 2023 --run-dir runs\essai_sasrec_2024_2023
.\.venv\Scripts\python.exe run.py evaluate-test --run-dir runs\essai_sasrec_2024_2023
.\.venv\Scripts\python.exe run.py baselines --split-seed 2024 --run-root runs\split2024
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

La fonction `resume` exige un checkpoint `last.pt` du même code et des mêmes données.
Ne modifier ni les JSON `data/processed`, ni les `splits/` une fois la campagne commencée.

## Licence et attribution

- **Code** : licence MIT, voir [`LICENSE`](LICENSE).
- **Données BIRD originales** : fournies par le projet MBANv2 / musée des Beaux-Arts de Nancy,
  sous licence **CC BY-NC-SA 4.0** (https://mbanv2.loria.fr/). Les fichiers de données dérivés
  de BIRD présents dans ce dépôt restent soumis à cette licence (pas d'usage commercial,
  partage dans les mêmes conditions).
- Si vous utilisez ces données, citez l'article original :
  > Alexanne Worm et al., *BIRD: A Museum Open Dataset Combining Behavior Patterns and
  > Identity Types to Better Model Visitors' Experience*. DOI : 10.1145/3708319.3733686

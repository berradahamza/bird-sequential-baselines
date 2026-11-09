# Méthode et protocole

Le problème est une classification du prochain POI dans le catalogue complet des 359
œuvres du musée, à partir d'un préfixe causal d'au plus 32 POI. Un préfixe par transition
adjacente, partitionnement strict des trajectoires au niveau du visiteur.

**Réseaux** : GRU4Rec adapté = embedding + GRU unidirectionnel + tête de scores ;
SASRec adapté = embeddings item/position, blocs d'attention causale, scores item-tied.
Entraînement par entropie croisée sur le prochain POI, AdamW. Il s'agit d'adaptations
« CE-prefix », pas d'implémentations de référence des auteurs.

**Modèles statistiques** : la popularité compte les *cibles* de chaque transition train.
Markov-1 compte (POI précédent -> cible), avec backoff vers la popularité. Markov-2
compte ((avant-dernier, dernier) -> cible), avec backoff Markov-1 puis popularité.
Lissage 0.001 sur les contextes vus, 0.001 par entrée pour la popularité, identique
à `STKG_CLL_POI/stkg_cll_poi/experiment.py::baseline_scores`.
Ces modèles n'ont pas de seed d'entraînement et sont calculés une fois par split.

**CLL** : splits 2022, 2023, 2024, 2025, 2026, chacun issu de `sorted(visitors)`,
`random.Random(split_seed).shuffle(ids)`, train = 35, validation = 8, test = 8.
Initialisations 2022, 2023, 2024 (réseaux seulement). Les partitions 2022..2025 ont
été comparées aux copies enregistrées dans l'archive CLL fournie ; celle de 2026 est
reconstruite par l'algorithme source (pas de split de run 2026 complet dans l'archive).

**Protocole fixe** : max 30 époques, patience 5, batch 64, hidden 64, dropout 0.2,
AdamW lr 0.001, weight decay 0.00005, historiques 32. Sélection uniquement sur
`validation.micro.mrr`, meilleur epoch sauvegardé dans `best.pt` ; test distinct,
jamais utilisé pour sélectionner. Les rangs sont calculés sur le catalogue complet,
avec tri déterministe des scores ex æquo. Rapport des métriques Acc@1, Hit@5,
Hit@10, MRR en micro, macro par visiteur, IC bootstrap en blocs visiteurs.

**Agrégation** : moyenne des 3 seeds par partition puis moyenne des 5 partitions ;
écart-type entre partitions descriptif. Pour les méthodes non entraînées : une seule
mesure par partition. Comparer exclusivement les résultats d'un même protocole BIRD.

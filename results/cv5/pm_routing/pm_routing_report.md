# PM routing interpretability: which pathways are kept vs dropped

**Cohort:** `pan_meta_pri` (mut_cnv) — 8893 samples, 1258 Reactome pathway tokens, 2 classes.  
**Protocol:** unified 5-fold CV, seed 42 (20% test / 10%-of-train val) — identical folds to Table 2. Model: canonical bioMoR (`bio_both`, learned pathway graph at both sites, sum-pooled).  
**Keep/drop signal:** each pathway token's *mean recursion depth* over the held-out test folds. Under **expert-choice** a capacity funnel keeps a shrinking top-$k$ each step, so depth $\in[0,K]$ counts how many steps the pathway survived; under **token-choice** each pathway self-gates one depth $\in[1,K]$. Higher depth = the router **keeps** allocating compute to that pathway; the minimum = it is **dropped** (exits early).

## Summary

| Routing | K | Macro-F1 | mean depth | active tokens per step | kept-to-K |
|---|---|---|---|---|---|
| expert | 1 | 85.1$\pm$1.3 | 1.00 | 1258 | 1258/1258 |
| expert | 2 | 84.7$\pm$2.8 | 1.75 | 1258, 944 | 1189/1258 |
| expert | 3 | 85.9$\pm$2.2 | 2.50 | 1258, 944, 944 | 684/1258 |
| expert | 4 | 83.9$\pm$1.9 | 3.25 | 1258, 944, 944, 944 | 379/1258 |
| token | 1 | 85.5$\pm$2.5 | 1.00 | 1258 | 1258/1258 |
| token | 2 | 86.1$\pm$1.5 | 1.49 | 1258, 611 | 604/1258 |
| token | 3 | 84.5$\pm$4.2 | 2.02 | 1258, 846, 436 | 115/1258 |
| token | 4 | 85.3$\pm$3.6 | 2.51 | 1258, 942, 629, 323 | 36/1258 |

## expert-choice, K=1  (Macro-F1 85.1$\pm$1.3)

_K=1 has a single pass, so there is no keep/drop decision (every pathway is kept to depth 1)._

## expert-choice, K=2  (Macro-F1 84.7$\pm$2.8)

**KEPT — top 25 deepest-routed pathways:**
1. `R-HSA-2559583` — depth 1.99
2. `R-HSA-2990846` — depth 1.99
3. `R-HSA-69563` — depth 1.99
4. `R-HSA-5358346` — depth 1.99
5. `R-HSA-163685` — depth 1.99
6. `R-HSA-157118` — depth 1.98
7. `R-HSA-5083625` — depth 1.98
8. `R-HSA-9755511` — depth 1.98
9. `R-HSA-2559580` — depth 1.98
10. `R-HSA-1236974` — depth 1.98
11. `R-HSA-9711123` — depth 1.98
12. `R-HSA-388841` — depth 1.98
13. `R-HSA-163560` — depth 1.98
14. `R-HSA-212165` — depth 1.98
15. `R-HSA-936964` — depth 1.98
16. `R-HSA-3000171` — depth 1.97
17. `R-HSA-5632684` — depth 1.97
18. `R-HSA-9917777` — depth 1.97
19. `R-HSA-5579029` — depth 1.97
20. `R-HSA-9759194` — depth 1.97
21. `R-HSA-1980145` — depth 1.97
22. `R-HSA-1989781` — depth 1.97
23. `R-HSA-9613829` — depth 1.97
24. `R-HSA-8982491` — depth 1.97
25. `R-HSA-9843745` — depth 1.97

**DROPPED — bottom 25 earliest-exit pathways:**
1. `R-HSA-9629569` — depth 1.09
2. `R-HSA-450513` — depth 1.20
3. `R-HSA-180024` — depth 1.22
4. `R-HSA-450385` — depth 1.27
5. `R-HSA-450604` — depth 1.31
6. `R-HSA-3560783` — depth 1.34
7. `R-HSA-210500` — depth 1.34
8. `R-HSA-392517` — depth 1.34
9. `R-HSA-110330` — depth 1.35
10. `R-HSA-450302` — depth 1.36
11. `R-HSA-5651801` — depth 1.37
12. `R-HSA-9920588` — depth 1.37
13. `R-HSA-9675135` — depth 1.38
14. `R-HSA-176187` — depth 1.38
15. `R-HSA-111471` — depth 1.38
16. `R-HSA-76071` — depth 1.38
17. `R-HSA-1810476` — depth 1.39
18. `R-HSA-5625740` — depth 1.39
19. `R-HSA-9709570` — depth 1.40
20. `R-HSA-844456` — depth 1.40
21. `R-HSA-5601884` — depth 1.41
22. `R-HSA-3560782` — depth 1.41
23. `R-HSA-9735871` — depth 1.41
24. `R-HSA-1606322` — depth 1.41
25. `R-HSA-114452` — depth 1.42

## expert-choice, K=3  (Macro-F1 85.9$\pm$2.2)

**KEPT — top 25 deepest-routed pathways:**
1. `R-HSA-3229121` — depth 2.99
2. `R-HSA-205043` — depth 2.99
3. `R-HSA-6803204` — depth 2.99
4. `R-HSA-6782315` — depth 2.98
5. `R-HSA-9764725` — depth 2.98
6. `R-HSA-6793080` — depth 2.98
7. `R-HSA-9920951` — depth 2.97
8. `R-HSA-9772755` — depth 2.97
9. `R-HSA-5654219` — depth 2.97
10. `R-HSA-9615710` — depth 2.97
11. `R-HSA-192823` — depth 2.97
12. `R-HSA-211976` — depth 2.97
13. `R-HSA-9619665` — depth 2.96
14. `R-HSA-9764560` — depth 2.96
15. `R-HSA-204998` — depth 2.96
16. `R-HSA-917729` — depth 2.96
17. `R-HSA-2173796` — depth 2.96
18. `R-HSA-196757` — depth 2.96
19. `R-HSA-1502540` — depth 2.95
20. `R-HSA-5083625` — depth 2.95
21. `R-HSA-8982491` — depth 2.95
22. `R-HSA-4641263` — depth 2.95
23. `R-HSA-6804757` — depth 2.95
24. `R-HSA-5633007` — depth 2.95
25. `R-HSA-9758274` — depth 2.94

**DROPPED — bottom 25 earliest-exit pathways:**
1. `R-HSA-3928664` — depth 1.52
2. `R-HSA-8862803` — depth 1.67
3. `R-HSA-3296482` — depth 1.68
4. `R-HSA-173623` — depth 1.69
5. `R-HSA-2514856` — depth 1.70
6. `R-HSA-5668599` — depth 1.73
7. `R-HSA-452723` — depth 1.74
8. `R-HSA-203927` — depth 1.75
9. `R-HSA-5627123` — depth 1.77
10. `R-HSA-4419969` — depth 1.77
11. `R-HSA-5601884` — depth 1.78
12. `R-HSA-1482788` — depth 1.78
13. `R-HSA-6803529` — depth 1.79
14. `R-HSA-1482839` — depth 1.80
15. `R-HSA-211000` — depth 1.82
16. `R-HSA-1482801` — depth 1.84
17. `R-HSA-9768727` — depth 1.84
18. `R-HSA-352230` — depth 1.87
19. `R-HSA-1482925` — depth 1.87
20. `R-HSA-5669034` — depth 1.87
21. `R-HSA-1482922` — depth 1.87
22. `R-HSA-73933` — depth 1.88
23. `R-HSA-606279` — depth 1.89
24. `R-HSA-450302` — depth 1.90
25. `R-HSA-9659787` — depth 1.90

## expert-choice, K=4  (Macro-F1 83.9$\pm$1.9)

**KEPT — top 25 deepest-routed pathways:**
1. `R-HSA-1632852` — depth 4.00
2. `R-HSA-2028269` — depth 3.99
3. `R-HSA-74751` — depth 3.99
4. `R-HSA-9663891` — depth 3.99
5. `R-HSA-2565942` — depth 3.99
6. `R-HSA-9678108` — depth 3.99
7. `R-HSA-9706574` — depth 3.99
8. `R-HSA-2404192` — depth 3.99
9. `R-HSA-442982` — depth 3.99
10. `R-HSA-8963899` — depth 3.99
11. `R-HSA-918233` — depth 3.99
12. `R-HSA-5628897` — depth 3.99
13. `R-HSA-3769402` — depth 3.98
14. `R-HSA-9013418` — depth 3.98
15. `R-HSA-9833110` — depth 3.98
16. `R-HSA-445989` — depth 3.98
17. `R-HSA-499943` — depth 3.98
18. `R-HSA-2559580` — depth 3.98
19. `R-HSA-168898` — depth 3.98
20. `R-HSA-109704` — depth 3.98
21. `R-HSA-9818027` — depth 3.98
22. `R-HSA-442742` — depth 3.98
23. `R-HSA-5693571` — depth 3.98
24. `R-HSA-1566977` — depth 3.98
25. `R-HSA-69206` — depth 3.98

**DROPPED — bottom 25 earliest-exit pathways:**
1. `R-HSA-977606` — depth 1.73
2. `R-HSA-166663` — depth 1.74
3. `R-HSA-166658` — depth 1.81
4. `R-HSA-202433` — depth 1.81
5. `R-HSA-2029481` — depth 1.84
6. `R-HSA-391160` — depth 1.94
7. `R-HSA-6803529` — depth 1.97
8. `R-HSA-1483206` — depth 1.99
9. `R-HSA-3371568` — depth 2.01
10. `R-HSA-9664323` — depth 2.05
11. `R-HSA-111465` — depth 2.19
12. `R-HSA-9828806` — depth 2.20
13. `R-HSA-9830364` — depth 2.21
14. `R-HSA-4419969` — depth 2.21
15. `R-HSA-202131` — depth 2.23
16. `R-HSA-1483166` — depth 2.24
17. `R-HSA-9820965` — depth 2.24
18. `R-HSA-2173788` — depth 2.24
19. `R-HSA-171319` — depth 2.24
20. `R-HSA-606279` — depth 2.25
21. `R-HSA-1482925` — depth 2.25
22. `R-HSA-8862803` — depth 2.25
23. `R-HSA-76071` — depth 2.25
24. `R-HSA-212300` — depth 2.25
25. `R-HSA-352230` — depth 2.26

## token-choice, K=1  (Macro-F1 85.5$\pm$2.5)

_K=1 has a single pass, so there is no keep/drop decision (every pathway is kept to depth 1)._

## token-choice, K=2  (Macro-F1 86.1$\pm$1.5)

**KEPT — top 25 deepest-routed pathways:**
1. `R-HSA-9920951` — depth 2.00
2. `R-HSA-9931510` — depth 1.99
3. `R-HSA-9821993` — depth 1.99
4. `R-HSA-379716` — depth 1.99
5. `R-HSA-75876` — depth 1.99
6. `R-HSA-140534` — depth 1.99
7. `R-HSA-9940465` — depth 1.99
8. `R-HSA-9648002` — depth 1.99
9. `R-HSA-9970672` — depth 1.99
10. `R-HSA-5357769` — depth 1.99
11. `R-HSA-912446` — depth 1.99
12. `R-HSA-5654228` — depth 1.98
13. `R-HSA-5654719` — depth 1.98
14. `R-HSA-5654712` — depth 1.98
15. `R-HSA-937072` — depth 1.98
16. `R-HSA-71336` — depth 1.98
17. `R-HSA-5690714` — depth 1.98
18. `R-HSA-428930` — depth 1.98
19. `R-HSA-982772` — depth 1.98
20. `R-HSA-9733458` — depth 1.98
21. `R-HSA-75105` — depth 1.98
22. `R-HSA-8849932` — depth 1.98
23. `R-HSA-9705462` — depth 1.97
24. `R-HSA-8875360` — depth 1.97
25. `R-HSA-392170` — depth 1.97

**DROPPED — bottom 25 earliest-exit pathways:**
1. `R-HSA-453274` — depth 1.00
2. `R-HSA-3858494` — depth 1.00
3. `R-HSA-170834` — depth 1.01
4. `R-HSA-9006936` — depth 1.01
5. `R-HSA-9010553` — depth 1.01
6. `R-HSA-195721` — depth 1.01
7. `R-HSA-4641258` — depth 1.01
8. `R-HSA-201681` — depth 1.01
9. `R-HSA-6791312` — depth 1.01
10. `R-HSA-2467813` — depth 1.01
11. `R-HSA-376176` — depth 1.02
12. `R-HSA-349425` — depth 1.02
13. `R-HSA-373755` — depth 1.02
14. `R-HSA-174178` — depth 1.02
15. `R-HSA-432722` — depth 1.02
16. `R-HSA-8873719` — depth 1.02
17. `R-HSA-9948299` — depth 1.02
18. `R-HSA-174143` — depth 1.02
19. `R-HSA-6811434` — depth 1.02
20. `R-HSA-72312` — depth 1.03
21. `R-HSA-450408` — depth 1.03
22. `R-HSA-9648025` — depth 1.03
23. `R-HSA-3371556` — depth 1.03
24. `R-HSA-162909` — depth 1.03
25. `R-HSA-3928665` — depth 1.03

## token-choice, K=3  (Macro-F1 84.5$\pm$4.2)

**KEPT — top 25 deepest-routed pathways:**
1. `R-HSA-9648002` — depth 2.99
2. `R-HSA-5358508` — depth 2.84
3. `R-HSA-9687136` — depth 2.81
4. `R-HSA-9709570` — depth 2.78
5. `R-HSA-9755779` — depth 2.78
6. `R-HSA-5689901` — depth 2.77
7. `R-HSA-6783310` — depth 2.77
8. `R-HSA-174048` — depth 2.76
9. `R-HSA-936964` — depth 2.75
10. `R-HSA-2243919` — depth 2.74
11. `R-HSA-6802946` — depth 2.74
12. `R-HSA-450321` — depth 2.74
13. `R-HSA-3000157` — depth 2.73
14. `R-HSA-9675126` — depth 2.72
15. `R-HSA-977225` — depth 2.71
16. `R-HSA-2214320` — depth 2.71
17. `R-HSA-442660` — depth 2.71
18. `R-HSA-176412` — depth 2.71
19. `R-HSA-3134975` — depth 2.70
20. `R-HSA-174437` — depth 2.70
21. `R-HSA-179409` — depth 2.70
22. `R-HSA-6805567` — depth 2.69
23. `R-HSA-6809371` — depth 2.69
24. `R-HSA-166663` — depth 2.68
25. `R-HSA-9823730` — depth 2.68

**DROPPED — bottom 25 earliest-exit pathways:**
1. `R-HSA-170834` — depth 1.01
2. `R-HSA-8853659` — depth 1.09
3. `R-HSA-5625900` — depth 1.12
4. `R-HSA-180292` — depth 1.14
5. `R-HSA-383280` — depth 1.14
6. `R-HSA-9006936` — depth 1.21
7. `R-HSA-5627123` — depth 1.22
8. `R-HSA-977444` — depth 1.23
9. `R-HSA-9665348` — depth 1.24
10. `R-HSA-5601884` — depth 1.24
11. `R-HSA-418038` — depth 1.25
12. `R-HSA-5617472` — depth 1.25
13. `R-HSA-2173789` — depth 1.25
14. `R-HSA-375280` — depth 1.26
15. `R-HSA-5610787` — depth 1.26
16. `R-HSA-8941326` — depth 1.26
17. `R-HSA-977443` — depth 1.27
18. `R-HSA-201681` — depth 1.28
19. `R-HSA-5578775` — depth 1.28
20. `R-HSA-9013404` — depth 1.29
21. `R-HSA-674695` — depth 1.29
22. `R-HSA-9006115` — depth 1.30
23. `R-HSA-167242` — depth 1.31
24. `R-HSA-418597` — depth 1.31
25. `R-HSA-451326` — depth 1.31

## token-choice, K=4  (Macro-F1 85.3$\pm$3.6)

**KEPT — top 25 deepest-routed pathways:**
1. `R-HSA-982772` — depth 3.82
2. `R-HSA-9828806` — depth 3.82
3. `R-HSA-9836573` — depth 3.80
4. `R-HSA-977068` — depth 3.76
5. `R-HSA-9820965` — depth 3.75
6. `R-HSA-9970672` — depth 3.74
7. `R-HSA-9821002` — depth 3.71
8. `R-HSA-9931295` — depth 3.71
9. `R-HSA-5334118` — depth 3.70
10. `R-HSA-937072` — depth 3.66
11. `R-HSA-8931838` — depth 3.66
12. `R-HSA-9710421` — depth 3.65
13. `R-HSA-196807` — depth 3.65
14. `R-HSA-199220` — depth 3.64
15. `R-HSA-3238698` — depth 3.64
16. `R-HSA-9821993` — depth 3.61
17. `R-HSA-187687` — depth 3.61
18. `R-HSA-1810476` — depth 3.61
19. `R-HSA-9619483` — depth 3.60
20. `R-HSA-9755779` — depth 3.58
21. `R-HSA-5635838` — depth 3.58
22. `R-HSA-1912420` — depth 3.58
23. `R-HSA-445144` — depth 3.58
24. `R-HSA-9037629` — depth 3.57
25. `R-HSA-70221` — depth 3.57

**DROPPED — bottom 25 earliest-exit pathways:**
1. `R-HSA-170834` — depth 1.28
2. `R-HSA-68949` — depth 1.36
3. `R-HSA-2565942` — depth 1.41
4. `R-HSA-373076` — depth 1.42
5. `R-HSA-9006936` — depth 1.42
6. `R-HSA-5673001` — depth 1.42
7. `R-HSA-5655302` — depth 1.44
8. `R-HSA-909733` — depth 1.44
9. `R-HSA-9637690` — depth 1.44
10. `R-HSA-69052` — depth 1.45
11. `R-HSA-1643713` — depth 1.46
12. `R-HSA-8878159` — depth 1.47
13. `R-HSA-2173793` — depth 1.49
14. `R-HSA-5633008` — depth 1.49
15. `R-HSA-5668541` — depth 1.50
16. `R-HSA-2559586` — depth 1.50
17. `R-HSA-9948299` — depth 1.52
18. `R-HSA-418038` — depth 1.54
19. `R-HSA-380320` — depth 1.55
20. `R-HSA-9820952` — depth 1.55
21. `R-HSA-9680350` — depth 1.56
22. `R-HSA-69306` — depth 1.57
23. `R-HSA-8864260` — depth 1.58
24. `R-HSA-1483257` — depth 1.58
25. `R-HSA-5683057` — depth 1.58

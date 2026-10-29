# 參數還原彙整

109 個複本。涵蓋率名目 89%；真相 = binom 時涵蓋率明顯低於 0.89 或偏差明顯非零，是模型自己的問題；真相 = betabinom 時的落差，是過度離散的代價。

## 每個模型一行

|                               |   參數數 |   平均絕對偏差 |   涵蓋率89 |   複本 |
|:------------------------------|------:|---------:|--------:|-----:|
| ('promoted_f4s', 'betabinom') |    22 |    0.009 |   0.827 |   20 |
| ('promoted_f4s', 'binom')     |    22 |    0.009 |   0.859 |   20 |
| ('promoted_f4sk', 'binom')    |    24 |    0.025 |   0.883 |    5 |
| ('promoted_f4so', 'binom')    |    26 |    0.011 |   0.862 |   10 |
| ('promoted_f5s', 'binom')     |    26 |    0.013 |   0.908 |   10 |
| ('promoted_f5sB', 'binom')    |    33 |    0.047 |   0.861 |    5 |
| ('promoted_f5sN', 'binom')    |    26 |    0.023 |   0.888 |   10 |
| ('promoted_f5sO', 'binom')    |    19 |    0.037 |   0.884 |    5 |
| ('war_f4s', 'binom')          |    31 |    0.046 |   0.869 |   20 |

## 每個參數

| 模型            | truth     | 參數               |     真值 |   平均估計 |   平均偏差 |   RMSE |   涵蓋率89 |   複本 |
|:--------------|:----------|:-----------------|-------:|-------:|-------:|-------:|--------:|-----:|
| promoted_f4s  | betabinom | λ AVG            |  0.084 |  0.084 |  0     |  0.005 |    0.8  |   20 |
| promoted_f4s  | betabinom | λ OBP            |  0.085 |  0.084 | -0     |  0.004 |    0.9  |   20 |
| promoted_f4s  | betabinom | λ K%             | -0.342 | -0.343 | -0.001 |  0.008 |    0.9  |   20 |
| promoted_f4s  | betabinom | λ whiff%         | -0.459 | -0.472 | -0.013 |  0.017 |    0.6  |   20 |
| promoted_f4s  | betabinom | λ XBH/H          |  0.388 |  0.389 |  0.001 |  0.013 |    0.85 |   20 |
| promoted_f4s  | betabinom | λ HR/AB          |  0.676 |  0.682 |  0.006 |  0.024 |    0.9  |   20 |
| promoted_f4s  | betabinom | λ Pull%          |  0.136 |  0.138 |  0.003 |  0.007 |    0.75 |   20 |
| promoted_f4s  | betabinom | λ GB%            | -0.253 | -0.261 | -0.008 |  0.01  |    0.7  |   20 |
| promoted_f4s  | betabinom | λ SB%            |  0.343 |  0.336 | -0.007 |  0.019 |    0.8  |   20 |
| promoted_f4s  | betabinom | λ SBA/OBT        |  0.988 |  0.99  |  0.002 |  0.019 |    0.95 |   20 |
| promoted_f4s  | betabinom | λ 3B/H           |  0.333 |  0.335 |  0.002 |  0.021 |    0.85 |   20 |
| promoted_f4s  | betabinom | λ GDP/AB         | -0.182 | -0.187 | -0.005 |  0.015 |    0.8  |   20 |
| promoted_f4s  | betabinom | λ BB%            |  0.389 |  0.392 |  0.003 |  0.012 |    0.85 |   20 |
| promoted_f4s  | betabinom | λ Ball%          |  0.288 |  0.298 |  0.01  |  0.012 |    0.5  |   20 |
| promoted_f4s  | betabinom | λ P/PA           |  0.094 |  0.093 | -0.001 |  0.014 |    0.8  |   20 |
| promoted_f4s  | betabinom | β contact        |  0.7   |  0.676 | -0.024 |  0.089 |    0.9  |   20 |
| promoted_f4s  | betabinom | β power          |  0.4   |  0.451 |  0.051 |  0.099 |    0.8  |   20 |
| promoted_f4s  | betabinom | β speed          |  0.1   |  0.125 |  0.025 |  0.091 |    0.9  |   20 |
| promoted_f4s  | betabinom | β discipline     | -0.3   | -0.305 | -0.005 |  0.077 |    0.95 |   20 |
| promoted_f4s  | betabinom | β_scout          |  0.4   |  0.398 | -0.002 |  0.068 |    1    |   20 |
| promoted_f4s  | betabinom | τ_org            |  0.3   |  0.285 | -0.015 |  0.087 |    0.9  |   20 |
| promoted_f4s  | betabinom | 因子相關（對角外平均）      |  0.3   |  0.29  | -0.01  |  0.03  |    0.8  |   20 |
| promoted_f4s  | binom     | λ AVG            |  0.084 |  0.083 | -0     |  0.004 |    0.85 |   20 |
| promoted_f4s  | binom     | λ OBP            |  0.085 |  0.085 |  0.001 |  0.003 |    0.9  |   20 |
| promoted_f4s  | binom     | λ K%             | -0.342 | -0.344 | -0.001 |  0.007 |    0.9  |   20 |
| promoted_f4s  | binom     | λ whiff%         | -0.459 | -0.462 | -0.003 |  0.009 |    0.9  |   20 |
| promoted_f4s  | binom     | λ XBH/H          |  0.388 |  0.388 | -0.001 |  0.012 |    0.9  |   20 |
| promoted_f4s  | binom     | λ HR/AB          |  0.676 |  0.682 |  0.005 |  0.021 |    0.95 |   20 |
| promoted_f4s  | binom     | λ Pull%          |  0.136 |  0.136 | -0     |  0.005 |    0.85 |   20 |
| promoted_f4s  | binom     | λ GB%            | -0.253 | -0.253 |  0     |  0.009 |    0.7  |   20 |
| promoted_f4s  | binom     | λ SB%            |  0.343 |  0.342 | -0.001 |  0.019 |    0.95 |   20 |
| promoted_f4s  | binom     | λ SBA/OBT        |  0.988 |  0.985 | -0.003 |  0.017 |    0.95 |   20 |
| promoted_f4s  | binom     | λ 3B/H           |  0.333 |  0.33  | -0.003 |  0.019 |    0.9  |   20 |
| promoted_f4s  | binom     | λ GDP/AB         | -0.182 | -0.182 |  0     |  0.013 |    0.9  |   20 |
| promoted_f4s  | binom     | λ BB%            |  0.389 |  0.393 |  0.004 |  0.011 |    0.8  |   20 |
| promoted_f4s  | binom     | λ Ball%          |  0.288 |  0.29  |  0.002 |  0.007 |    0.9  |   20 |
| promoted_f4s  | binom     | λ P/PA           |  0.094 |  0.094 |  0.001 |  0.013 |    0.8  |   20 |
| promoted_f4s  | binom     | β contact        |  0.7   |  0.734 |  0.034 |  0.091 |    0.9  |   20 |
| promoted_f4s  | binom     | β power          |  0.4   |  0.393 | -0.007 |  0.109 |    0.8  |   20 |
| promoted_f4s  | binom     | β speed          |  0.1   |  0.067 | -0.033 |  0.075 |    0.85 |   20 |
| promoted_f4s  | binom     | β discipline     | -0.3   | -0.329 | -0.029 |  0.08  |    0.85 |   20 |
| promoted_f4s  | binom     | β_scout          |  0.4   |  0.438 |  0.038 |  0.135 |    0.7  |   20 |
| promoted_f4s  | binom     | τ_org            |  0.3   |  0.272 | -0.028 |  0.115 |    0.75 |   20 |
| promoted_f4s  | binom     | 因子相關（對角外平均）      |  0.3   |  0.295 | -0.005 |  0.027 |    0.9  |   20 |
| promoted_f4sk | binom     | λ AVG            |  0.4   |  0.404 |  0.004 |  0.005 |    1    |    5 |
| promoted_f4sk | binom     | λ OBP            |  0.4   |  0.403 |  0.003 |  0.007 |    1    |    5 |
| promoted_f4sk | binom     | λ K%             | -0.4   | -0.404 | -0.004 |  0.008 |    1    |    5 |
| promoted_f4sk | binom     | λ whiff%         | -0.4   | -0.403 | -0.003 |  0.006 |    1    |    5 |
| promoted_f4sk | binom     | λ XBH/H          |  0.4   |  0.394 | -0.006 |  0.01  |    1    |    5 |
| promoted_f4sk | binom     | λ HR/AB          |  0.4   |  0.382 | -0.018 |  0.027 |    0.8  |    5 |
| promoted_f4sk | binom     | λ Pull%          |  0.4   |  0.395 | -0.005 |  0.009 |    0.8  |    5 |
| promoted_f4sk | binom     | λ GB%            | -0.4   | -0.396 |  0.004 |  0.006 |    1    |    5 |
| promoted_f4sk | binom     | λ SB%            |  0.4   |  0.411 |  0.011 |  0.024 |    0.8  |    5 |
| promoted_f4sk | binom     | λ SBA/OBT        |  0.4   |  0.399 | -0.001 |  0.01  |    1    |    5 |
| promoted_f4sk | binom     | λ 3B/H           |  0.4   |  0.407 |  0.007 |  0.067 |    0.8  |    5 |
| promoted_f4sk | binom     | λ GDP/AB         | -0.4   | -0.402 | -0.002 |  0.011 |    1    |    5 |
| promoted_f4sk | binom     | λ BB%            |  0.4   |  0.404 |  0.004 |  0.01  |    0.8  |    5 |
| promoted_f4sk | binom     | λ Ball%          |  0.4   |  0.402 |  0.002 |  0.005 |    1    |    5 |
| promoted_f4sk | binom     | λ P/PA           |  0.4   |  0.401 |  0.001 |  0.01  |    1    |    5 |
| promoted_f4sk | binom     | β contact        |  0.7   |  0.723 |  0.023 |  0.132 |    0.6  |    5 |
| promoted_f4sk | binom     | β power          |  0.4   |  0.409 |  0.009 |  0.039 |    1    |    5 |
| promoted_f4sk | binom     | β speed          |  0.1   |  0.148 |  0.048 |  0.171 |    0.8  |    5 |
| promoted_f4sk | binom     | β discipline     | -0.3   | -0.353 | -0.053 |  0.054 |    1    |    5 |
| promoted_f4sk | binom     | β_scout          |  0.4   |  0.456 |  0.056 |  0.162 |    0.4  |    5 |
| promoted_f4sk | binom     | β 型態×母隊強弱        | -0.3   | -0.011 |  0.289 |  0.305 |    0.6  |    5 |
| promoted_f4sk | binom     | σ_org 型態         |  0.15  |  0.138 | -0.012 |  0.062 |    1    |    5 |
| promoted_f4sk | binom     | τ_org            |  0.3   |  0.317 |  0.017 |  0.053 |    1    |    5 |
| promoted_f4sk | binom     | 因子相關（對角外平均）      |  0.3   |  0.31  |  0.01  |  0.03  |    0.8  |    5 |
| promoted_f4so | binom     | λ AVG            |  0.084 |  0.084 | -0     |  0.003 |    0.9  |   10 |
| promoted_f4so | binom     | λ OBP            |  0.085 |  0.084 | -0     |  0.003 |    1    |   10 |
| promoted_f4so | binom     | λ K%             | -0.342 | -0.342 |  0     |  0.007 |    0.9  |   10 |
| promoted_f4so | binom     | λ whiff%         | -0.458 | -0.459 | -0.001 |  0.009 |    0.9  |   10 |
| promoted_f4so | binom     | λ XBH/H          |  0.388 |  0.386 | -0.002 |  0.011 |    0.8  |   10 |
| promoted_f4so | binom     | λ HR/AB          |  0.675 |  0.685 |  0.01  |  0.025 |    0.8  |   10 |
| promoted_f4so | binom     | λ Pull%          |  0.136 |  0.135 | -0.001 |  0.007 |    0.6  |   10 |
| promoted_f4so | binom     | λ GB%            | -0.253 | -0.257 | -0.003 |  0.009 |    0.7  |   10 |
| promoted_f4so | binom     | λ SB%            |  0.343 |  0.342 | -0.001 |  0.015 |    1    |   10 |
| promoted_f4so | binom     | λ SBA/OBT        |  0.988 |  0.992 |  0.004 |  0.014 |    1    |   10 |
| promoted_f4so | binom     | λ 3B/H           |  0.332 |  0.339 |  0.007 |  0.03  |    0.6  |   10 |
| promoted_f4so | binom     | λ GDP/AB         | -0.182 | -0.187 | -0.005 |  0.017 |    0.7  |   10 |
| promoted_f4so | binom     | λ BB%            |  0.388 |  0.387 | -0.001 |  0.007 |    1    |   10 |
| promoted_f4so | binom     | λ Ball%          |  0.288 |  0.288 |  0     |  0.006 |    0.9  |   10 |
| promoted_f4so | binom     | λ P/PA           |  0.094 |  0.095 |  0.002 |  0.013 |    0.8  |   10 |
| promoted_f4so | binom     | β contact        |  0.7   |  0.719 |  0.019 |  0.126 |    0.7  |   10 |
| promoted_f4so | binom     | β power          |  0.4   |  0.419 |  0.019 |  0.091 |    0.8  |   10 |
| promoted_f4so | binom     | β speed          |  0.1   |  0.097 | -0.003 |  0.085 |    0.9  |   10 |
| promoted_f4so | binom     | β discipline     | -0.3   | -0.301 | -0.001 |  0.063 |    1    |   10 |
| promoted_f4so | binom     | β_scout          |  0.4   |  0.48  |  0.08  |  0.123 |    0.9  |   10 |
| promoted_f4so | binom     | σ_org contact    |  0.15  |  0.214 |  0.064 |  0.106 |    0.9  |   10 |
| promoted_f4so | binom     | σ_org power      |  0.15  |  0.163 |  0.013 |  0.061 |    1    |   10 |
| promoted_f4so | binom     | σ_org speed      |  0.15  |  0.146 | -0.004 |  0.051 |    1    |   10 |
| promoted_f4so | binom     | σ_org discipline |  0.15  |  0.186 |  0.036 |  0.096 |    0.9  |   10 |
| promoted_f4so | binom     | τ_org            |  0.3   |  0.303 |  0.003 |  0.12  |    0.8  |   10 |
| promoted_f4so | binom     | 因子相關（對角外平均）      |  0.3   |  0.311 |  0.011 |  0.025 |    0.9  |   10 |
| promoted_f5s  | binom     | λ AVG            |  0.084 |  0.081 | -0.003 |  0.005 |    0.8  |   10 |
| promoted_f5s  | binom     | λ OBP            |  0.085 |  0.085 | -0     |  0.003 |    1    |   10 |
| promoted_f5s  | binom     | λ K%             | -0.343 | -0.342 |  0.001 |  0.007 |    1    |   10 |
| promoted_f5s  | binom     | λ whiff%         | -0.459 | -0.458 |  0.002 |  0.007 |    1    |   10 |
| promoted_f5s  | binom     | λ XBH/H          |  0.389 |  0.39  |  0.001 |  0.008 |    1    |   10 |
| promoted_f5s  | binom     | λ HR/AB          |  0.679 |  0.68  |  0.001 |  0.023 |    0.9  |   10 |
| promoted_f5s  | binom     | λ Pull%          |  0.135 |  0.132 | -0.003 |  0.005 |    0.9  |   10 |
| promoted_f5s  | binom     | λ GB%            | -0.254 | -0.254 | -0     |  0.005 |    1    |   10 |
| promoted_f5s  | binom     | λ SB%            |  0.342 |  0.345 |  0.003 |  0.014 |    1    |   10 |
| promoted_f5s  | binom     | λ SBA/OBT        |  0.986 |  0.979 | -0.006 |  0.026 |    0.8  |   10 |
| promoted_f5s  | binom     | λ 3B/H           |  0.331 |  0.33  | -0.001 |  0.016 |    0.9  |   10 |
| promoted_f5s  | binom     | λ GDP/AB         | -0.182 | -0.182 | -0     |  0.009 |    1    |   10 |
| promoted_f5s  | binom     | λ BB%            |  0.389 |  0.395 |  0.006 |  0.012 |    0.7  |   10 |
| promoted_f5s  | binom     | λ Ball%          |  0.288 |  0.291 |  0.003 |  0.006 |    0.9  |   10 |
| promoted_f5s  | binom     | λ P/PA           |  0.094 |  0.099 |  0.005 |  0.013 |    0.9  |   10 |
| promoted_f5s  | binom     | λ LD%            |  0.14  |  0.145 |  0.006 |  0.009 |    0.8  |   10 |
| promoted_f5s  | binom     | λ IFFB%          | -0.289 | -0.294 | -0.005 |  0.012 |    0.8  |   10 |
| promoted_f5s  | binom     | λ BABIP          |  0.117 |  0.117 | -0     |  0.004 |    1    |   10 |
| promoted_f5s  | binom     | β contact        |  0.7   |  0.79  |  0.09  |  0.115 |    0.8  |   10 |
| promoted_f5s  | binom     | β power          |  0.4   |  0.412 |  0.012 |  0.079 |    0.9  |   10 |
| promoted_f5s  | binom     | β speed          |  0.1   |  0.146 |  0.046 |  0.092 |    0.9  |   10 |
| promoted_f5s  | binom     | β discipline     | -0.3   | -0.372 | -0.072 |  0.108 |    0.9  |   10 |
| promoted_f5s  | binom     | β quality        |  0.3   |  0.291 | -0.009 |  0.068 |    0.9  |   10 |
| promoted_f5s  | binom     | β_scout          |  0.4   |  0.444 |  0.044 |  0.128 |    0.9  |   10 |
| promoted_f5s  | binom     | τ_org            |  0.3   |  0.275 | -0.025 |  0.087 |    1    |   10 |
| promoted_f5s  | binom     | 因子相關（對角外平均）      |  0.3   |  0.299 | -0.001 |  0.026 |    0.9  |   10 |
| war_f4s       | binom     | λ AVG            |  0.084 |  0.084 | -0     |  0.005 |    0.8  |   20 |
| war_f4s       | binom     | λ OBP            |  0.085 |  0.085 |  0.001 |  0.003 |    0.9  |   20 |
| war_f4s       | binom     | λ K%             | -0.342 | -0.344 | -0.002 |  0.007 |    0.9  |   20 |
| war_f4s       | binom     | λ whiff%         | -0.459 | -0.462 | -0.004 |  0.009 |    0.9  |   20 |
| war_f4s       | binom     | λ XBH/H          |  0.388 |  0.388 | -0.001 |  0.011 |    0.9  |   20 |
| war_f4s       | binom     | λ HR/AB          |  0.676 |  0.682 |  0.006 |  0.021 |    0.95 |   20 |
| war_f4s       | binom     | λ Pull%          |  0.136 |  0.136 | -0     |  0.005 |    0.9  |   20 |
| war_f4s       | binom     | λ GB%            | -0.253 | -0.254 | -0     |  0.009 |    0.7  |   20 |
| war_f4s       | binom     | λ SB%            |  0.343 |  0.342 | -0.001 |  0.019 |    0.95 |   20 |
| war_f4s       | binom     | λ SBA/OBT        |  0.988 |  0.985 | -0.003 |  0.018 |    0.95 |   20 |
| war_f4s       | binom     | λ 3B/H           |  0.333 |  0.33  | -0.002 |  0.019 |    0.9  |   20 |
| war_f4s       | binom     | λ GDP/AB         | -0.182 | -0.182 |  0.001 |  0.013 |    0.85 |   20 |
| war_f4s       | binom     | λ BB%            |  0.389 |  0.393 |  0.004 |  0.011 |    0.75 |   20 |
| war_f4s       | binom     | λ Ball%          |  0.288 |  0.29  |  0.002 |  0.007 |    0.85 |   20 |
| war_f4s       | binom     | λ P/PA           |  0.094 |  0.095 |  0.001 |  0.013 |    0.8  |   20 |
| war_f4s       | binom     | t_β contact      | -0.4   | -0.396 |  0.004 |  0.027 |    0.85 |   20 |
| war_f4s       | binom     | t_β power        | -0.2   | -0.207 | -0.007 |  0.026 |    0.85 |   20 |
| war_f4s       | binom     | t_β speed        | -0.1   | -0.093 |  0.007 |  0.026 |    0.85 |   20 |
| war_f4s       | binom     | t_β discipline   |  0.2   |  0.204 |  0.004 |  0.029 |    0.8  |   20 |
| war_f4s       | binom     | t_β_scout        | -0.3   | -0.294 |  0.006 |  0.029 |    0.95 |   20 |
| war_f4s       | binom     | w_β contact      |  0.3   |  0.342 |  0.042 |  0.06  |    0.8  |   20 |
| war_f4s       | binom     | w_β power        |  0.4   |  0.413 |  0.013 |  0.049 |    0.9  |   20 |
| war_f4s       | binom     | w_β speed        |  0     | -0.018 | -0.018 |  0.051 |    0.95 |   20 |
| war_f4s       | binom     | w_β discipline   |  0.1   |  0.083 | -0.017 |  0.052 |    0.9  |   20 |
| war_f4s       | binom     | w_β_scout        |  0.1   |  0.138 |  0.038 |  0.081 |    0.95 |   20 |
| war_f4s       | binom     | t_τ_org          |  0.3   |  0.297 | -0.003 |  0.048 |    0.85 |   20 |
| war_f4s       | binom     | w_τ_org          |  0.3   |  0.267 | -0.033 |  0.082 |    0.9  |   20 |
| war_f4s       | binom     | σ_t              |  0.6   |  0.6   |  0     |  0.022 |    0.75 |   20 |
| war_f4s       | binom     | σ_w              |  1     |  1.017 |  0.017 |  0.045 |    0.95 |   20 |
| war_f4s       | binom     | ν                |  5     |  6.197 |  1.197 |  1.579 |    0.8  |   20 |
| war_f4s       | binom     | 因子相關（對角外平均）      |  0.3   |  0.295 | -0.005 |  0.026 |    0.9  |   20 |
| promoted_f5sB | binom     | λ AVG            |  0.4   |  0.409 |  0.009 |  0.01  |    1    |    5 |
| promoted_f5sB | binom     | λ OBP            |  0.4   |  0.406 |  0.006 |  0.009 |    1    |    5 |
| promoted_f5sB | binom     | λ K%             | -0.4   | -0.404 | -0.004 |  0.008 |    1    |    5 |
| promoted_f5sB | binom     | λ whiff%         | -0.4   | -0.405 | -0.005 |  0.007 |    1    |    5 |
| promoted_f5sB | binom     | λ XBH/H          |  0.4   |  0.391 | -0.009 |  0.021 |    0.6  |    5 |
| promoted_f5sB | binom     | λ HR/AB          |  0.4   |  0.377 | -0.023 |  0.029 |    0.8  |    5 |
| promoted_f5sB | binom     | λ Pull%          |  0.4   |  0.396 | -0.004 |  0.015 |    1    |    5 |
| promoted_f5sB | binom     | λ GB%            | -0.4   | -0.387 |  0.013 |  0.022 |    0.6  |    5 |
| promoted_f5sB | binom     | λ SB%            |  0.4   |  0.387 | -0.013 |  0.023 |    1    |    5 |
| promoted_f5sB | binom     | λ SBA/OBT        |  0.4   |  0.388 | -0.012 |  0.017 |    1    |    5 |
| promoted_f5sB | binom     | λ 3B/H           |  0.4   |  0.392 | -0.008 |  0.024 |    1    |    5 |
| promoted_f5sB | binom     | λ GDP/AB         | -0.4   | -0.393 |  0.007 |  0.015 |    1    |    5 |
| promoted_f5sB | binom     | λ BB%            |  0.4   |  0.397 | -0.003 |  0.014 |    0.8  |    5 |
| promoted_f5sB | binom     | λ Ball%          |  0.4   |  0.397 | -0.003 |  0.014 |    0.8  |    5 |
| promoted_f5sB | binom     | λ P/PA           |  0.4   |  0.395 | -0.005 |  0.019 |    0.8  |    5 |
| promoted_f5sB | binom     | λ LD%            |  0.4   |  0.406 |  0.006 |  0.017 |    0.8  |    5 |
| promoted_f5sB | binom     | λ IFFB%          | -0.4   | -0.417 | -0.017 |  0.019 |    0.8  |    5 |
| promoted_f5sB | binom     | λ BABIP          |  0.4   |  0.403 |  0.003 |  0.01  |    1    |    5 |
| promoted_f5sB | binom     | λ Z-Con%         |  0.4   |  0.404 |  0.004 |  0.007 |    1    |    5 |
| promoted_f5sB | binom     | λ EV90           |  0.4   |  0.373 | -0.027 |  0.037 |    0.6  |    5 |
| promoted_f5sB | binom     | λ Barrel%        |  0.4   |  0.388 | -0.012 |  0.02  |    0.8  |    5 |
| promoted_f5sB | binom     | λ PullAir%       |  0.4   |  0.388 | -0.012 |  0.019 |    0.6  |    5 |
| promoted_f5sB | binom     | λ O-Swing%       | -0.4   | -0.4   | -0     |  0.014 |    0.8  |    5 |
| promoted_f5sB | binom     | λ EV10           |  0.4   |  0.415 |  0.015 |  0.043 |    0.6  |    5 |
| promoted_f5sB | binom     | λ xwOBAcon       |  0.4   |  0.425 |  0.025 |  0.03  |    0.8  |    5 |
| promoted_f5sB | binom     | β contact        |  0.7   |  0.966 |  0.266 |  0.303 |    0.6  |    5 |
| promoted_f5sB | binom     | β power          |  0.4   |  0.588 |  0.188 |  0.213 |    1    |    5 |
| promoted_f5sB | binom     | β speed          |  0.1   |  0.168 |  0.068 |  0.146 |    1    |    5 |
| promoted_f5sB | binom     | β discipline     | -0.3   | -0.481 | -0.181 |  0.227 |    0.8  |    5 |
| promoted_f5sB | binom     | β quality        |  0.3   |  0.58  |  0.28  |  0.336 |    0.8  |    5 |
| promoted_f5sB | binom     | β_scout          |  0.4   |  0.541 |  0.141 |  0.212 |    1    |    5 |
| promoted_f5sB | binom     | τ_org            |  0.3   |  0.458 |  0.158 |  0.234 |    1    |    5 |
| promoted_f5sB | binom     | 因子相關（對角外平均）      |  0.3   |  0.283 | -0.017 |  0.032 |    1    |    5 |
| promoted_f5sN | binom     | λ AVG            |  0.4   |  0.403 |  0.003 |  0.01  |    1    |   10 |
| promoted_f5sN | binom     | λ OBP            |  0.4   |  0.403 |  0.003 |  0.008 |    1    |   10 |
| promoted_f5sN | binom     | λ K%             | -0.4   | -0.401 | -0.001 |  0.009 |    1    |   10 |
| promoted_f5sN | binom     | λ whiff%         | -0.4   | -0.403 | -0.003 |  0.008 |    1    |   10 |
| promoted_f5sN | binom     | λ XBH/H          |  0.4   |  0.4   |  0     |  0.018 |    0.8  |   10 |
| promoted_f5sN | binom     | λ HR/AB          |  0.4   |  0.395 | -0.005 |  0.027 |    0.9  |   10 |
| promoted_f5sN | binom     | λ Pull%          |  0.4   |  0.4   | -0     |  0.014 |    0.9  |   10 |
| promoted_f5sN | binom     | λ GB%            | -0.4   | -0.396 |  0.004 |  0.019 |    0.8  |   10 |
| promoted_f5sN | binom     | λ SB%            |  0.4   |  0.399 | -0.001 |  0.025 |    1    |   10 |
| promoted_f5sN | binom     | λ SBA/OBT        |  0.4   |  0.396 | -0.004 |  0.016 |    1    |   10 |
| promoted_f5sN | binom     | λ 3B/H           |  0.4   |  0.396 | -0.004 |  0.038 |    0.8  |   10 |
| promoted_f5sN | binom     | λ GDP/AB         | -0.4   | -0.398 |  0.002 |  0.024 |    0.9  |   10 |
| promoted_f5sN | binom     | λ BB%            |  0.4   |  0.402 |  0.002 |  0.013 |    0.9  |   10 |
| promoted_f5sN | binom     | λ Ball%          |  0.4   |  0.402 |  0.002 |  0.013 |    0.9  |   10 |
| promoted_f5sN | binom     | λ P/PA           |  0.4   |  0.403 |  0.003 |  0.02  |    0.8  |   10 |
| promoted_f5sN | binom     | λ LD%            |  0.4   |  0.406 |  0.006 |  0.015 |    0.9  |   10 |
| promoted_f5sN | binom     | λ IFFB%          | -0.4   | -0.412 | -0.012 |  0.022 |    0.8  |   10 |
| promoted_f5sN | binom     | λ BABIP          |  0.4   |  0.402 |  0.002 |  0.011 |    0.9  |   10 |
| promoted_f5sN | binom     | β contact        |  0.7   |  0.849 |  0.149 |  0.231 |    0.8  |   10 |
| promoted_f5sN | binom     | β power          |  0.4   |  0.465 |  0.065 |  0.172 |    0.9  |   10 |
| promoted_f5sN | binom     | β speed          |  0.1   |  0.046 | -0.054 |  0.271 |    0.6  |   10 |
| promoted_f5sN | binom     | β discipline     | -0.3   | -0.396 | -0.096 |  0.243 |    0.9  |   10 |
| promoted_f5sN | binom     | β quality        |  0.3   |  0.379 |  0.079 |  0.215 |    0.8  |   10 |
| promoted_f5sN | binom     | β_scout          |  0.4   |  0.444 |  0.044 |  0.255 |    0.8  |   10 |
| promoted_f5sN | binom     | τ_org            |  0.3   |  0.353 |  0.053 |  0.122 |    1    |   10 |
| promoted_f5sN | binom     | 因子相關（對角外平均）      |  0.3   |  0.296 | -0.004 |  0.036 |    1    |   10 |
| promoted_f5sO | binom     | λ SB%            |  0.4   |  0.458 |  0.058 |  0.064 |    0.6  |    5 |
| promoted_f5sO | binom     | λ SBA/OBT        |  0.4   |  0.401 |  0.001 |  0.014 |    1    |    5 |
| promoted_f5sO | binom     | λ 3B/H           |  0.4   |  0.38  | -0.02  |  0.024 |    1    |    5 |
| promoted_f5sO | binom     | λ GDP/AB         | -0.4   | -0.407 | -0.007 |  0.014 |    1    |    5 |
| promoted_f5sO | binom     | λ Z-Con%         |  0.4   |  0.404 |  0.004 |  0.01  |    1    |    5 |
| promoted_f5sO | binom     | λ EV90           |  0.4   |  0.392 | -0.008 |  0.024 |    0.8  |    5 |
| promoted_f5sO | binom     | λ Barrel%        |  0.4   |  0.394 | -0.006 |  0.015 |    0.8  |    5 |
| promoted_f5sO | binom     | λ PullAir%       |  0.4   |  0.391 | -0.009 |  0.026 |    0.6  |    5 |
| promoted_f5sO | binom     | λ O-Swing%       | -0.4   | -0.398 |  0.002 |  0.014 |    0.8  |    5 |
| promoted_f5sO | binom     | λ EV10           |  0.4   |  0.413 |  0.013 |  0.037 |    0.8  |    5 |
| promoted_f5sO | binom     | λ xwOBAcon       |  0.4   |  0.417 |  0.017 |  0.033 |    1    |    5 |
| promoted_f5sO | binom     | β contact        |  0.7   |  0.649 | -0.051 |  0.114 |    1    |    5 |
| promoted_f5sO | binom     | β power          |  0.4   |  0.611 |  0.211 |  0.375 |    0.8  |    5 |
| promoted_f5sO | binom     | β speed          |  0.1   |  0.137 |  0.037 |  0.204 |    1    |    5 |
| promoted_f5sO | binom     | β discipline     | -0.3   | -0.323 | -0.023 |  0.221 |    0.8  |    5 |
| promoted_f5sO | binom     | β quality        |  0.3   |  0.369 |  0.069 |  0.151 |    1    |    5 |
| promoted_f5sO | binom     | β_scout          |  0.4   |  0.506 |  0.106 |  0.236 |    0.8  |    5 |
| promoted_f5sO | binom     | τ_org            |  0.3   |  0.338 |  0.038 |  0.078 |    1    |    5 |
| promoted_f5sO | binom     | 因子相關（對角外平均）      |  0.3   |  0.281 | -0.019 |  0.032 |    1    |    5 |

## η、u_org（與斜率偏離）與真值的相關，r-hat

|                               |   η 相關 contact |   η 相關 power |   η 相關 speed |   η 相關 discipline |   u_org 相關 |   r-hat |   型態偏離相關 |   斜率偏離相關 |   η 相關 quality |
|:------------------------------|---------------:|-------------:|-------------:|------------------:|-----------:|--------:|---------:|---------:|---------------:|
| ('promoted_f4s', 'betabinom') |          0.954 |        0.895 |        0.904 |             0.922 |      0.635 |   1.074 |  nan     |  nan     |        nan     |
| ('promoted_f4s', 'binom')     |          0.964 |        0.905 |        0.907 |             0.939 |      0.623 |   1.056 |  nan     |  nan     |        nan     |
| ('promoted_f4sk', 'binom')    |          0.977 |        0.948 |        0.823 |             0.958 |      0.557 |   1.672 |    0.021 |  nan     |        nan     |
| ('promoted_f4so', 'binom')    |          0.964 |        0.903 |        0.907 |             0.939 |      0.604 |   1.09  |  nan     |    0.349 |        nan     |
| ('promoted_f5s', 'binom')     |          0.963 |        0.902 |        0.91  |             0.939 |      0.669 |   1.059 |  nan     |  nan     |          0.806 |
| ('promoted_f5sB', 'binom')    |          0.98  |        0.955 |        0.831 |             0.979 |      0.299 |   1.087 |  nan     |  nan     |          0.932 |
| ('promoted_f5sN', 'binom')    |          0.977 |        0.945 |        0.841 |             0.971 |      0.297 |   1.053 |  nan     |  nan     |          0.929 |
| ('promoted_f5sO', 'binom')    |          0.921 |        0.876 |        0.827 |             0.942 |      0.361 |   1.024 |  nan     |  nan     |          0.71  |
| ('war_f4s', 'binom')          |          0.966 |        0.907 |        0.908 |             0.94  |      0.93  |   1.064 |  nan     |  nan     |        nan     |

# Sistema de Busca e Ranking de Produtos (WANDS)

**Autor:** Carlos Eduardo Barbosa

> *Nota: Código e documentação desenvolvidos por Carlos Eduardo Barbosa com suporte do Gemini (Google AI), de forma interativa por meio de engenharia de prompts, refinamento arquitetural e validação prática no dataset WANDS.*

## Visão Geral
Projeto de busca e ordenação de produtos de ponta a ponta construído sobre o dataset **WANDS (Wayfair Annotation Dataset)**. A arquitetura opera em dois estágios, com parâmetros centralizados em `config.yaml` (`src/config.py`):
1. **Retrieval (Recuperação):** utiliza `BM25Retriever` para recuperar os `top_k` candidatos (padrão `top_k=100`) a partir do catálogo de produtos.
2. **Ranking (Ordenação):** utiliza `HistGradientBoostingRegressor` (`scikit-learn`, instanciado via `build_ranker` em `src/ranking.py`) para reordenar os candidatos combinando `28` features de recuperação/sobreposição lexical e atributos do catálogo/consulta, com suporte nativo a valores ausentes (`NaN`).

---

## Estrutura do Projeto

```text
.
├── artifacts/
│   └── hgb_ranker.joblib                   # Modelo treinado (28 features) e lista de colunas utilizadas
├── config.yaml                             # Configuração centralizada (seeds, splits, top_k, limites e modelo)
├── data/
│   ├── raw/                                # Arquivos originais do WANDS separados por tabulação (\t)
│   └── processed/                          # Camada tratada em Parquet (products, queries, labels, splits)
├── notebooks/
│   ├── 00_initial_eda.ipynb                # Análise exploratória inicial do catálogo, consultas e rótulos WANDS
│   ├── 01_retrieval_experiments.ipynb      # Experimentos de recuperação lexical e avaliação de Recall@K (BM25)
│   ├── 02_ranking_experiments.ipynb        # Engenharia de features, estudo de ablação (val) e teste one-shot                            # EDA, experimentos e avaliação no dataset WANDS
│   └── 03_latency_and_error_analysis.ipynb  # Benchmark de latência, indexação e análise de erros
├── src/
│   ├── config.py                           # Leitura e validação da configuração centralizada
│   ├── paths.py                            # Centralização de caminhos relativos do projeto
│   ├── data.py                             # Leitura bruta, saneamento e materialização da camada tratada
│   ├── split.py                            # Divisão determinística agrupada por query_id (train/val/test)
│   ├── validate_query.py                   # Validação e saneamento de consultas de entrada (vazias/longas)
│   ├── retrieval.py                        # Recuperação lexical com BM25 e indexação do catálogo
│   ├── features.py                         # Engenharia de 28 features de recuperação, sobreposição e catálogo
│   ├── ranking.py                          # Fábrica de modelos, treinamento, carregamento e inferência
│   ├── evaluation.py                       # Métricas de recuperação (Recall@K) e ranking (NDCG@K)
│   ├── pipeline.py                         # Orquestração treino/avaliação e busca ponta a ponta
│   └── cli.py                              # Interface de linha de comando (CLI)
├── tests/
│   └── test_search_ranking.py              # Testes unitários e de integração com pytest
├── README.md
└── requirements.txt
```

---

## Configuração do Ambiente

Crie e ative um ambiente virtual e instale as dependências:

python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt

Para executar a suíte de testes:

python -m pytest -v

---

## Execução Completa do Zero (Ordem Recomendada)

Para reproduzir o ambiente, os artefatos e os resultados a partir do zero:

1. **Ambiente e testes automatizados:** crie o ambiente virtual, instale as dependências de `requirements.txt` e rode `python -m pytest -v` para validar métricas, splits, retrieval e ranking.
2. **Dados brutos (WANDS):** coloque `product.csv`, `query.csv` e `label.csv` (delimitados por `\t`) em `data/raw/`.
3. **Preparação e divisão (`prepare` -> `split`):** execute `python -m src.cli prepare` para gerar `data/processed/` e `python -m src.cli split` para criar o split determinístico agrupado por `query_id` (`train`, `val` e `test` sem vazamento).
4. **Treinamento e validação (`train`):** execute `python -m src.cli train --top-k 100` para treinar o `HistGradientBoostingRegressor` em `train`, avaliar em `val` (`Recall@K` e `NDCG@10`) e gerar/atualizar `artifacts/hgb_ranker.joblib`.
5. **Busca via CLI (`search`):** execute `python -m src.cli search "modern coffee table" --top-k 10` para testar a validação e a ordenação ponta a ponta.
6. **Avaliação one-shot no teste, latência, indexação e erros (opcional / reprodução completa):** execute os notebooks em `notebooks/` (incluindo `03_latency_and_error_analysis.ipynb`) para reproduzir a avaliação única em `test`, a medição de latência (`p50`/`p95`), o custo de indexação offline e a análise qualitativa de erros.

## Dados e Preparação (Camada Raw -> Processed)

1. **Obtenção dos dados brutos:** coloque os três arquivos originais do WANDS (`product.csv`, `query.csv` e `label.csv`, separados por `\t`) dentro de `data/raw/`.
2. **Materialização da camada tratada (`src/data.py`):**

python -m src.cli prepare

3. **Divisão determinística por consulta (`src/split.py`):** gera o mapeamento `query_id -> split` (`train` / `val` / `test`) em `data/processed/splits.parquet` agrupado por `query_id`, garantindo que todas as interações de uma mesma consulta fiquem em um único conjunto e evitando vazamento entre treino, validação e teste:

python -m src.cli split

### Tratamentos aplicados na etapa `prepare`
- **Padronização de esquema:** ajuste de `category hierarchy` para `category_hierarchy` e tipagem explícita de identificadores (`product_id` e `query_id` como `int64`).
- **Saneamento de texto:** substituição de valores nulos por strings vazias (`""`) e normalização de quebras de linha e espaços excedentes sem gerar tokens `"nan"`.
- **Atributos numéricos e rótulos:** conversão numérica segura (`rating_count`, `average_rating`, `review_count`) preservando `NaN` para tratamento nativo em modelos baseados em árvores, mapeamento de relevância graduada e deduplicação por chave.

---

## Uso via CLI (Preparação, Divisão, Treino e Busca)

A CLI (`src/cli.py`) concentra os comandos do pipeline; abaixo está a sequência operacional após `prepare`:

### 1. Dividir consultas em `train`, `val` e `test` (`src/split.py`)
Particiona os `query_id`s de forma determinística e salva `data/processed/splits.parquet`, mantendo todas as linhas de cada consulta no mesmo split para evitar vazamento:

python -m src.cli split

### 2. Treinar e avaliar o pipeline
Executa a recuperação de candidatos (`BM25Retriever`), a construção das `28` features, o treino do `HistGradientBoostingRegressor` em `train`, a avaliação em `val` (`Recall@K` e `NDCG@10`) e salva o artefato em `artifacts/hgb_ranker.joblib`:

python -m src.cli train --top-k 100

### 3. Buscar produtos e recuperar resultados ranqueados
Valida a consulta em `src/validate_query.py` (rejeitando consultas vazias ou apenas espaços e truncando consultas acima do limite de caracteres configurado) e executa a busca ponta a ponta exibindo `id`, `product_name` e `score`:

python -m src.cli search "modern coffee table" --top-k 10

---

## Protocolo de Avaliação

- **Split agrupado por `query_id` (`src/split.py`):** garante que todas as avaliações de uma mesma consulta pertençam a um único conjunto (`train`, `val` ou `test`).
- **Rótulos graduados:** mapeamento das avaliações do WANDS para relevância numérica (`Exact = 2`, `Partial = 1`, `Irrelevant / não avaliado = 0`).
- **Métricas:**
  - **Estágio 1 (Retrieval):** `Recall@20`, `Recall@50` e `Recall@100` sobre o conjunto de candidatos recuperados (`top_k=100`).
  - **Estágio 2 (Ranking):** `NDCG@10` comparando a ordenação base do `BM25` contra a reordenação do `HistGradientBoostingRegressor`.

---

## Retrieval e Ranking: Experimentos e Ablação (`val`)

Na validação (`72` consultas, `7.200` pares candidato-consulta com `top_k=100`), o `BM25Retriever` obteve `Recall@20 = 0.1243`, `Recall@50 = 0.2509` e `Recall@100 = 0.3831`, com `NDCG@10` inicial de `0.7103`.

### Escolha do Modelo de Ranking
O estágio de ranking foi padronizado com `sklearn.ensemble.HistGradientBoostingRegressor` (`28` features), escolhido pela eficiência do treinamento/inferência baseado em histogramas, suporte nativo a valores ausentes (`NaN`) sem imputação artificial e ganho consistente de `NDCG@10`.

### Ablação por Grupo de Features na Validação (`val`)

| Configuração / Grupo de Features | Qtd. Features | `NDCG@10` (`val`) | Ganho vs. BM25 |
| :--- | :---: | :---: | :---: |
| BM25 baseline (apenas recuperação) | 0 | 0.7103 | — |
| Features de recuperação e correspondência lexical | 12 | 0.7675 | +0.0572 |
| Features de atributos do catálogo, tamanho e cobertura | 16 | 0.7686 | +0.0583 |
| **Todas as features combinadas (`hgb_ranker.joblib`)** | **28** | **0.7755** | **+0.0653** |

---

## Resultados Finais no Teste, Latência e Análise de Erros

### Avaliação Única no Conjunto de Teste (`test`)
Após selecionar a configuração de `28` features na validação, o modelo salvo foi avaliado **uma única vez** em `test` (`72` consultas, `7.200` candidatos com `top_k=100`), registrando `Recall@20 = 0.0969`, `Recall@50 = 0.2308` e `Recall@100 = 0.3662` no retrieval:

| Conjunto | Consultas | Candidatos (`top_k=100`) | BM25 `NDCG@10` | HistGradientBoosting `NDCG@10` | Ganho (`Δ`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `val` | 72 | 7.200 | 0.7103 | 0.7755 | +0.0653 |
| **`test` (one-shot)** | **72** | **7.200** | **0.6706** | **0.7380** | **+0.0674** |

### Latência por Consulta e Custo de Indexação (`03_latency_and_error_analysis.ipynb`)

| Etapa | `p50` (ms) | `p95` (ms) | Média (ms) |
| :--- | :---: | :---: | :---: |
| 1. BM25 Retrieval | 146.5 | 880.0 | 316.2 |
| 2. Extração de Features + HGB Scoring | 89.9 | 151.8 | 103.2 |
| **Total End-to-End (por consulta)** | **266.3** | **960.4** | **419.4** |

- **Custo de indexação offline:** a construção do índice BM25 sobre os `42.994` produtos do catálogo levou `65,0 s` (mediana) / `85,2 s` (média), com pico de `816,7 MB` de memória em Python, sendo executada de forma assíncrona fora da inferência online.

### Análise de Erros e Limitações
- **Desempenho por consulta:** no conjunto de teste (`72` consultas), o `HistGradientBoostingRegressor` superou o `BM25` em **38** consultas, empatou em **20** e ficou abaixo em **14**.
- **Dependência da busca inicial:** em **3 de 72** consultas (`"star wars rug"`, `"small loving roomtables"` e `"minnestrista"`), nenhum item relevante entrou no `top_k=100`, limitando o `NDCG@10` do ranker a `0.0000`.
- **Rótulos incompletos e consultas amplas:** em buscas genéricas (ex.: `"bedroom accessories"`), o dataset avalia apenas uma amostra do catálogo, podendo penalizar itens relevantes promovidos pelo modelo sem anotação no gabarito.
- **Sobreposição de termos e typos:** consultas com erros de grafia (ex.: `"pedistole sink"`, onde o ranker elevou o `NDCG@10` de `0.0000` para `0.1850`) ou termos compartilhados com acessórios/peças (ex.: `"46 inch closet door"`) evidenciam os limites da correspondência puramente lexical sem sinais comportamentais online (cliques, navegação e compras).
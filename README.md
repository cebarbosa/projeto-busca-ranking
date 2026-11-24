# Sistema de Busca e Ranking de Produtos (WANDS)

## Visão Geral
Projeto de busca de produtos de ponta a ponta construído sobre o dataset **WANDS (Wayfair Annotation Dataset)**. A arquitetura separa o problema em dois estágios principais para permitir avaliação independente e escalabilidade:
1. **Retrieval (Recuperação):** filtra candidatos relevantes a partir do catálogo completo de produtos.
2. **Ranking (Ordenação):** reordena os candidatos recuperados combinando sinais textuais/categóricos e atributos numéricos do produto.

---

## Estrutura do Projeto

```text
├── data/
│   ├── raw/                  # Arquivos originais do WANDS separados por tabulação (\t)
│   └── processed/            # Camada tratada em Parquet (tipagem e limpeza básica)
├── notebooks/                # Notebooks para EDA, experimentos e análise de erros
├── src/
│   ├── paths.py              # Centralização de caminhos relativos do projeto
│   ├── data.py               # Leitura bruta, saneamento e materialização da camada tratada
│   └── cli.py                # Interface de linha de comando (CLI)
├── README.md
└── requirements.txt


## Configuração do Ambiente

Crie e ative um ambiente virtual e instale as dependências:

```bash
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Dados e Preparação (Camada Raw -> Processed)

1. **Obtenção dos dados brutos:** coloque os três arquivos originais do WANDS (`product.csv`, `query.csv` e `label.csv`, separados por `\t`) dentro do diretório `data/raw/`.
2. **Materialização da camada tratada:** execute o comando abaixo na raiz do repositório para ler os arquivos brutos, aplicar o saneamento básico e salvar as tabelas tratadas em `data/processed/` (formato `.parquet`):

```bash
python -m src.cli prepare
```

### Tratamentos aplicados na etapa `prepare`
* **Padronização de esquema:** correção do nome da coluna `category hierarchy` para `category_hierarchy` e tipagem explícita de identificadores (`product_id` e `query_id` como `int64`) para garantir integridade nos cruzamentos.
* **Saneamento de texto:** substituição de valores nulos por strings vazias (`""`) e normalização de quebras de linha/espaços excedentes, evitando tokens artificiais (`"nan"`) sem perder capitalização original.
* **Atributos numéricos e rótulos:** conversão segura de colunas numéricas (`rating_count`, `average_rating`, `review_count`) para `float64` (preservando `NaN` para tratamento nativo em modelos baseados em árvores), padronização dos rótulos (`Exact`, `Partial`, `Irrelevant`) e remoção de duplicatas por chave.

## Uso via CLI

Além da preparação de dados, a CLI expõe o ponto de entrada para consulta (com validação de entradas inválidas, como consultas vazias ou excessivamente longas):

```bash
python -m src.cli search "modern coffee table" --top-k 10
```

## Protocolo de Avaliação
*(Em desenvolvimento: definição da divisão treino/validação/teste agrupada por `query_id` para evitar vazamento de dados, mapeamento de relevância graduada/binária e escolha de métricas para retrieval e ranking.)*

## Retrieval: Baseline Lexical vs. Abordagem Semântica/Híbrida
*(Em desenvolvimento: comparação experimental na validação e justificativa da abordagem selecionada.)*

## Ranking: Engenharia de Features e Modelo
*(Em desenvolvimento: descrição das features textuais, numéricas e relativas à query, construção dos exemplos e seleção do modelo na validação.)*

## Resultados Finais no Teste, Limitações e Próximos Passos
*(Em desenvolvimento: avaliação única no conjunto de teste, análise de erros e considerações para produção.)*
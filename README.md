# Prática 3 — Soma dos quadrados com memória distribuída (MPI)

Programa em **Python + mpi4py** que calcula a soma dos quadrados dos números de `0` até `n-1`
dividindo o trabalho entre vários processos MPI. A prática demonstra o modelo de **memória distribuída**:
cada processo possui seus próprios dados locais e a troca de informação é feita apenas por mensagens.

> Disciplina: Programação de Aplicações de Alto Desempenho — Engenharia da Computação — Universidade Veiga de Almeida (UVA)
> Aluno: Vítor Peyrek Cardoso

## Como funciona

1. O **rank 0** (raiz) cria a lista completa `range(n)` e a divide em uma sublista por processo (distribuição **cíclica**: `lista[rank::total_processos]`).
2. `scatter()` entrega a cada processo **somente a sua sublista**.
3. Cada processo calcula a sua **soma parcial** de forma independente.
4. `gather()` recolhe no rank 0 as tuplas `(rank, dados_locais, soma_parcial)`.
5. O rank 0 soma as parciais e confere o resultado com o **cálculo sequencial** e com a **fórmula fechada** `(n-1)·n·(2n-1)/6`.

Para `[0, 1, 2, 3, 4, 5]` com 3 processos: rank 0 → `[0, 3]`, rank 1 → `[1, 4]`, rank 2 → `[2, 5]`.
A divisão cíclica não perde elementos quando `n` não é múltiplo do número de processos e mantém a carga equilibrada (diferença máxima de 1 elemento entre ranks).

## Requisitos

- Python 3.8+
- Uma implementação MPI instalada:
  - **Windows:** [Microsoft MPI (MS-MPI)](https://learn.microsoft.com/message-passing-interface/microsoft-mpi) — instale o *Runtime* e o *SDK*
  - **Linux (Debian/Ubuntu):** `sudo apt install openmpi-bin libopenmpi-dev`
  - **macOS:** `brew install open-mpi`
- Biblioteca `mpi4py`:

```bash
pip install -r requirements.txt
```

## Execução

```bash
mpiexec -n 4 python Pratica3.py --n 20
```

| Argumento | Descrição | Padrão |
|-----------|-----------|--------|
| `--n`   | Quantidade de números (0 até n-1). Deve ser ≥ 0. | 16 |
| `-n` (mpiexec) | Número de processos MPI. | — |

> Em máquinas com poucos núcleos, o OpenMPI pode exigir `mpiexec --oversubscribe -n 4 ...`.

## Exemplo de saída

`mpiexec -n 4 python Pratica3.py --n 20`

```text
[RAIZ] 20 números; 4 processos MPI.

[RESULTADOS POR PROCESSO]
  rank 0: dados=[0, 4, 8, 12, 16]; soma parcial=480
  rank 1: dados=[1, 5, 9, 13, 17]; soma parcial=565
  rank 2: dados=[2, 6, 10, 14, 18]; soma parcial=660
  rank 3: dados=[3, 7, 11, 15, 19]; soma parcial=765

[RESULTADO DISTRIBUÍDO]  2470
[CONFERÊNCIA SEQUENCIAL] 2470
[CONFERÊNCIA FÓRMULA]    2470
[CORRETO?] True
[TEMPO PARALELO (s)]   0.011220
[TEMPO SEQUENCIAL (s)] 0.000005
[OBS.] Tempo inclui a comunicação MPI e varia a cada execução; não use como benchmark.
```

Conferência manual: `19·20·39/6 = 2470`. A saída deve exibir `[CORRETO?] True`.

## Observações sobre desempenho

Com `n` pequeno o custo de comunicação (`scatter`/`gather` e serialização com *pickle*) domina o tempo total,
então a versão paralela **não** é mais rápida que a sequencial. O ganho só aparece quando o custo de computação
por processo supera o custo de comunicação (`n` muito grande ou cálculo mais pesado). Os tempos impressos variam a
cada execução e **não** devem ser usados como benchmark.

## Arquivos

| Arquivo | Descrição |
|---------|-----------|
| `Pratica3.py` | Código-fonte da prática |
| `requirements.txt` | Dependências Python |
| `README.md` | Este arquivo |

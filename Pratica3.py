"""
Prática 3 - Soma dos quadrados com memória distribuída (MPI / mpi4py).

Objetivo
--------
Mostrar que processos MPI possuem dados locais separados (memória
distribuída) e cooperam para calcular a soma dos quadrados dos números
de 0 até n-1.

Fluxo
-----
1. O rank 0 (raiz) cria a lista completa e a divide em partes (cíclico).
2. scatter() entrega a cada processo SOMENTE a sua parte.
3. Cada processo calcula a sua soma parcial de forma independente.
4. gather() devolve ao rank 0 as tuplas (rank, dados, parcial).
5. O rank 0 soma as parciais e confere com o cálculo sequencial e com a
   fórmula fechada (n-1)·n·(2n-1)/6.

Execução
--------
    mpiexec -n 4 python Pratica3.py --n 20
"""

import argparse

try:
    from mpi4py import MPI
except ImportError as erro:
    raise SystemExit(
        "A biblioteca mpi4py não foi encontrada.\n"
        "Instale com:  pip install mpi4py\n"
        "e verifique se há uma implementação MPI (MS-MPI, OpenMPI ou MPICH)."
    ) from erro


def particionar_ciclico(sequencia, num_processos):
    """Divide 'sequencia' em uma sublista por processo (distribuição cíclica).

    Exemplo: [0..5] com 3 processos -> [[0, 3], [1, 4], [2, 5]].
    Não perde elementos quando n não é divisível pelo número de processos.
    """
    return [sequencia[indice::num_processos] for indice in range(num_processos)]


def soma_de_quadrados(valores):
    """Calcula SOMENTE a parcela que pertence a quem a executa."""
    return sum(valor * valor for valor in valores)


def formula_fechada(n):
    """Soma dos quadrados de 0 até n-1: (n-1)·n·(2n-1)/6."""
    return (n - 1) * n * (2 * n - 1) // 6


def ler_argumentos():
    parser = argparse.ArgumentParser(
        description="Soma de quadrados com memória distribuída usando MPI."
    )
    parser.add_argument(
        "--n", type=int, default=16,
        help="Quantidade de números: 0 até n-1 (padrão: 16).",
    )
    args = parser.parse_args()
    if args.n < 0:
        parser.error("--n deve ser maior ou igual a zero")
    return args


def main():
    args = ler_argumentos()

    comunicador = MPI.COMM_WORLD
    rank = comunicador.Get_rank()
    total_processos = comunicador.Get_size()

    # Somente a raiz conhece a lista completa; os demais ainda não têm dados.
    if rank == 0:
        lista_completa = list(range(args.n))
        partes = particionar_ciclico(lista_completa, total_processos)
        print(f"\n[RAIZ] {args.n} números; {total_processos} processos MPI.",
              flush=True)
    else:
        partes = None

    # Alinha todos os processos antes de medir o tempo.
    comunicador.Barrier()
    inicio = MPI.Wtime()

    # Cada processo recebe apenas a sua sublista (dados locais separados).
    dados_locais = comunicador.scatter(partes, root=0)

    # Cálculo independente em cada processo.
    parcial_local = soma_de_quadrados(dados_locais)

    # A raiz recolhe (rank, dados locais, soma parcial) de todos.
    coletado = comunicador.gather((rank, dados_locais, parcial_local), root=0)

    if rank == 0:
        tempo_paralelo = MPI.Wtime() - inicio

        print("\n[RESULTADOS POR PROCESSO]")
        for rank_origem, dados, parcial in sorted(coletado, key=lambda t: t[0]):
            print(f"  rank {rank_origem}: dados={dados}; soma parcial={parcial}")

        total_distribuido = sum(parcial for _, _, parcial in coletado)

        t0 = MPI.Wtime()
        total_sequencial = soma_de_quadrados(range(args.n))
        tempo_sequencial = MPI.Wtime() - t0

        total_formula = formula_fechada(args.n)

        print(f"\n[RESULTADO DISTRIBUÍDO]  {total_distribuido}")
        print(f"[CONFERÊNCIA SEQUENCIAL] {total_sequencial}")
        print(f"[CONFERÊNCIA FÓRMULA]    {total_formula}")
        print(f"[CORRETO?] "
              f"{total_distribuido == total_sequencial == total_formula}")
        print(f"[TEMPO PARALELO (s)]   {tempo_paralelo:.6f}")
        print(f"[TEMPO SEQUENCIAL (s)] {tempo_sequencial:.6f}")
        print("[OBS.] Tempo inclui a comunicação MPI e varia a cada execução; "
              "não use como benchmark.\n")


if __name__ == "__main__":
    main()

def sgemm_nn_heur_even_m(args):
    return args["m"] % args["BLOCK_M"] == 0


def sgemm_nn_heur_even_n(args):
    return args["n"] % args["BLOCK_N"] == 0


def sgemm_nn_heur_even_k(args):
    return args["k"] % args["BLOCK_K"] == 0


HEURISTICS_CONFIGS = {
    "sgemm_nn": {
        "EVEN_M": sgemm_nn_heur_even_m,
        "EVEN_N": sgemm_nn_heur_even_n,
        "EVEN_K": sgemm_nn_heur_even_k,
    },
}

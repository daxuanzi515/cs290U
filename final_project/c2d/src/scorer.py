import torch


def score_candidates(results, alpha=0.5, beta=0.5,
                     Tmin=0.3, Cmin=0.5):
    # results = [{x_cf, T, C}]
    T_vals = torch.tensor([r["T"] for r in results])
    C_vals = torch.tensor([r["C"] for r in results])

    T_norm = (T_vals - T_vals.min()) / (T_vals.max() - T_vals.min() + 1e-8)
    C_norm = (C_vals - C_vals.min()) / (C_vals.max() - C_vals.min() + 1e-8)

    best = None
    best_score = -1

    for i, r in enumerate(results):
        if T_norm[i] < Tmin or C_norm[i] < Cmin:
            continue
        
        score = (T_norm[i] ** alpha) * (C_norm[i] ** beta)
        if score > best_score:
            best_score = score
            best = r["x_cf"]

    return best, best_score

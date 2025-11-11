# ===== IA-RFE with XGBoost GPUTreeShap + Pearson→(gated)DeLong =====
# Only compute DeLong when Pearson p-value (r_p) >= 0.05.

import numpy as np
import pandas as pd
import copy, json
from tqdm import tqdm

import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve, auc, average_precision_score
# from scipy.stats import pearsonr  # removed: no longer gating by Pearson

# ---- knobs ----
ALPHA_INTERACT = 0.05
ROW_CAP        = 3254

# ---- helper: compute network importance from GPUTreeShap interaction tensor ----
def _gpu_network_importance(model, X_df, alpha=ALPHA_INTERACT, row_cap=ROW_CAP):
    if (row_cap is not None) and (len(X_df) > row_cap):
        X_used = X_df.sample(row_cap, random_state=42)
    else:
        X_used = X_df
    dmat = xgb.DMatrix(X_used)
    booster = model.get_booster()
    inter = booster.predict(dmat, pred_interactions=True)   # (n,F(+1),F(+1))
    M = np.mean(np.abs(inter), axis=0)

    # strip bias row/col if present
    F_feat = X_df.shape[1]
    if M.shape[0] == F_feat + 1 and M.shape[1] == F_feat + 1:
        M = M[:-1, :-1]   # (F,F)

    # adaptive alpha (optional)
    diag_vals = np.diag(M)
    M_off = M.copy()
    np.fill_diagonal(M_off, np.nan)
    off_med = np.nanmedian(M_off)
    diag_med = np.median(diag_vals)
    ratio   = off_med / (diag_med + 1e-12)
    alpha = 1.5 * ratio
    # alpha   = float(np.clip(1.0 * ratio, 0.05, 0.4))

    main = np.diag(M)
    total_pair = M.sum(axis=1) - main
    NI = main + alpha * total_pair
    return NI, M

# ---- lightweight CV to get concatenated probs (no bootstrap) ----
def _cv_concat_probs(X_df, y, features, kf, device='cuda'):
    """Return concatenated out-of-fold probabilities for a given feature set."""
    probs, labs = [], []
    for tr_idx, va_idx in kf.split(X_df[features], y):
        X_tr, X_va = X_df.iloc[tr_idx][features], X_df.iloc[va_idx][features]
        y_tr, y_va = y[tr_idx], y[va_idx]
        mdl = xgb.XGBClassifier(device=device)
        mdl.fit(X_tr, y_tr)
        probs.append(mdl.predict_proba(X_va)[:, 1])
        labs.append(y_va)
    return np.concatenate(probs), np.concatenate(labs)

# ================== your original variables ==================
model = xgb.XGBClassifier(device='cuda')
pivot_table_new_reduced = copy.deepcopy(pivot_table_new)
top_features = list(pivot_table_new_reduced.columns)

top_features_total = []
p_value_acc = []          # DeLong p per iteration
effect_size_acc = []
auc_acc = []
y_prob_428 = np.concatenate(delong_y_prob_428)   # baseline probs (fixed)
y_label    = np.concatenate(delong_y_label)
auc_lower = []
auc_upper = []
auc_std = []
delta_auc = []

# ================== IA-RFE outer loop ==================
for j in tqdm(range(pivot_table_new.shape[1] - 1)):
    model = xgb.XGBClassifier(device='cuda')

    kf = KFold(n_splits=5, shuffle=False)
    X = pivot_table_new_reduced[top_features]
    y = expire_flags

    top_features_total.append(top_features.copy())

    tprs = []
    aucs = []
    aucs_lower_round = []
    aucs_upper_round = []
    mean_fpr = np.linspace(0, 1, 100)
    std_auc_single = []
    delta_auc_single = []

    delong_y_prob_323 = []  # kept (not used for DeLong anymore)
    precision = []
    TPR = []

    # -------- CV loop (metrics only; DeLong removed here) --------
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        delong_y_prob_323.append(y_prob)
        precision.append(average_precision_score(y_test, y_prob))

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        for i in range(len(fpr)):
            if fpr[i] > 0.05:
                TPR.append(tpr[i]); break
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # --- bootstrap for CI (reduced iterations) ---
        n_bootstraps = 200
        bootstrapped_scores = []
        rng = np.random.RandomState(42)
        for _ in range(n_bootstraps):
            idx = rng.randint(0, len(y_prob), len(y_prob))
            if len(np.unique(y_test[idx])) < 2:
                continue
            bfpr, btpr, _ = roc_curve(y_test[idx], y_prob[idx])
            for k in range(len(bfpr)):
                if bfpr[k] > 0.05:
                    TPR.append(btpr[k]); break
            bootstrapped_scores.append(auc(bfpr, btpr))
        sorted_scores = np.array(bootstrapped_scores)
        std_auc_single.append(np.std(sorted_scores))
        delta_auc_single.append(sorted_scores)
        sorted_scores.sort()
        aucs_lower_round.append(sorted_scores[int(0.025 * len(sorted_scores))])
        aucs_upper_round.append(sorted_scores[int(0.975 * len(sorted_scores))])

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    # aggregate fold metrics
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    auc_acc.append(mean_auc)
    auc_std.append(np.mean(std_auc_single))
    delta_auc.append(np.mean(np.array(delta_auc_single), axis=0))
    auc_lower.append(np.mean(aucs_lower_round))
    auc_upper.append(np.mean(aucs_upper_round))

    # ---- xgboost base_score safeguard (kept) ----
    bst = model.get_booster()
    cfg = json.loads(bst.save_config())
    bs = cfg["learner"]["learner_model_param"]["base_score"]
    if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
        cfg["learner"]["learner_model_param"]["base_score"] = str(float(bs.strip("[]")))
        bst.load_config(json.dumps(cfg))

    # ===================== IA-RFE: SHAP-interaction → Network Importance =====================
    full_model = xgb.XGBClassifier(device='cuda')
    full_model.fit(X, y)
    NI, M = _gpu_network_importance(full_model, X, alpha=ALPHA_INTERACT, row_cap=ROW_CAP)

    # ---- candidate order (worst NI → less-worst) ----
    sorted_idx = np.argsort(NI)

    chosen_idx = None
    chosen_feat = None
    y_prob_after_del = None
    final_p = np.nan          # DeLong p for chosen deletion
    final_effect_size = np.nan

    # Upward search without Pearson gating: always compute DeLong
    for cand_idx in sorted_idx:
        cand_feat = top_features[int(cand_idx)]
        feat_after = [f for f in top_features if f != cand_feat]

        # OOF probs after removing this candidate
        y_prob_cand, _ = _cv_concat_probs(pivot_table_new_reduced, y, feat_after, kf, device='cuda')

        # Always compute DeLong between baseline and candidate
        p_log10_cand, delta_auc_def, z_effect_def = delong_roc_test(y_label, y_prob_428, y_prob_cand)
        p_cand = 10**p_log10_cand
        try:
            p_cand_scalar = float(np.asarray(p_cand)[0][0])
        except Exception:
            p_cand_scalar = float(np.asarray(p_cand).squeeze())
        # choose this candidate if not significantly different by DeLong
        if p_cand_scalar >= 0.05:
            chosen_idx = int(cand_idx)
            chosen_feat = cand_feat
            y_prob_after_del = y_prob_cand
            final_p = p_cand_scalar
            final_effect_size = z_effect_def
            break
        else:
            # continue searching for a safer candidate
            continue

    # Fallback: if no candidate passed the rules, delete the worst NI
    if chosen_idx is None:
        worst_idx = int(sorted_idx[0])
        chosen_idx = worst_idx
        chosen_feat = top_features[worst_idx]
        feat_after_default = [f for f in top_features if f != chosen_feat]
        y_prob_after_del, _ = _cv_concat_probs(pivot_table_new_reduced, y, feat_after_default, kf, device='cuda')
        p_log10_def, delta_auc_def, z_effect_def = delong_roc_test(y_label, y_prob_428, y_prob_after_del)
        p_def = 10**p_log10_def
        try:
            final_p = float(np.asarray(p_def)[0][0])
        except Exception:
            final_p = float(np.asarray(p_def).squeeze())
        final_effect_size = z_effect_def

    # ---- record and remove chosen feature ----
    p_value_acc.append(final_p)
    effect_size_acc.append(final_effect_size)

    top_features.pop(chosen_idx)

# -------- outputs:
# - top_features_total, auc_acc, auc_lower, auc_upper, auc_std, delta_auc
# - p_value_acc (DeLong p)

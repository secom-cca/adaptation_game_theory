#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

# ================================================================
# (1) sim_data の作成：CSV ファイル "sim_data.csv" を読み込む
# ================================================================
# CSV ファイルは、各行が以下の形式になっているとする：
# Forest,Dam,Embankment,Relocation,Run,upstream_cost,downstream_cost,cumulative_damage,final_ecosystem
# ここで、Forest, Dam, Embankment, Relocation は "True" または "False" の文字列
# 試行結果は数値として出力されている
sim_data = {}
csv_filename = "sim_data.csv"
with open(csv_filename, "r", newline="") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        # 各行から、Forest, Dam, Embankment, Relocation の値を取得（文字列 "True"/"False" を bool に変換）
        forest_bool = (row["Forest"].strip().lower() == "true")
        dam_bool    = (row["Dam"].strip().lower() == "true")
        embank_bool = (row["Embankment"].strip().lower() == "true")
        reloc_bool  = (row["Relocation"].strip().lower() == "true")
        # 上流戦略の決定
        if forest_bool and dam_bool:
            u_strat = "Both"
        elif forest_bool:
            u_strat = "Forest"
        elif dam_bool:
            u_strat = "Dam"
        else:
            u_strat = "No"
        # 下流戦略の決定
        if embank_bool and reloc_bool:
            d_strat = "Both"
        elif embank_bool:
            d_strat = "Embankment"
        elif reloc_bool:
            d_strat = "Relocation"
        else:
            d_strat = "No"
        # 試行番号は row["Run"]（数値として扱う）
        run = int(row["Run"])
        # 結果は各数値カラムから float 値として取得
        up_cost = float(row["upstream_cost"])
        down_cost = float(row["downstream_cost"])
        flood_dmg = float(row["cumulative_damage"])
        ecosys = float(row["final_ecosystem"])
        
        key = (u_strat, d_strat)
        if key not in sim_data:
            sim_data[key] = {
                "upstream_cost": [],
                "downstream_cost": [],
                "flood_damage": [],
                "ecosystem": []
            }
        sim_data[key]["upstream_cost"].append(up_cost)
        sim_data[key]["downstream_cost"].append(down_cost)
        sim_data[key]["flood_damage"].append(flood_dmg)
        sim_data[key]["ecosystem"].append(ecosys)

# 各リストを numpy 配列に変換
for key in sim_data:
    sim_data[key]["upstream_cost"] = np.array(sim_data[key]["upstream_cost"])
    sim_data[key]["downstream_cost"] = np.array(sim_data[key]["downstream_cost"])
    sim_data[key]["flood_damage"] = np.array(sim_data[key]["flood_damage"])
    sim_data[key]["ecosystem"] = np.array(sim_data[key]["ecosystem"])

print(f"Loaded simulation data from {csv_filename}")

# ================================================================
# (2) Risk attitude parameters (choose one of 5, 50, 95)
#     These determine which percentile of the simulation distribution is used.
# ================================================================
risk_percentile_U = 50  # Upstream (U): e.g. 5 (risk-averse), 50 (risk-neutral), 95 (risk-seeking)
risk_percentile_D = 99  # Downstream (D)

# ================================================================
# (3) Extract raw component values (per strategy pair) from simulation data
#     For U: econ_U = -upstream_cost, safe_U = -flood_damage, env_U = ecosystem
#     For D: econ_D = -downstream_cost, safe_D = -flood_damage, env_D = ecosystem
# ================================================================
raw_payoffs_U = {}  # Key: (u_strat, d_strat), Value: dict {"econ", "safe", "env"}
raw_payoffs_D = {}

# strategies_U, strategies_D は以下の通りとする
strategies_U = ["No", "Forest", "Dam", "Both"]
strategies_D = ["No", "Embankment", "Relocation", "Both"]

for u in strategies_U:
    for d in strategies_D:
        key = (u, d)
        if key in sim_data:
            econ_U_val = -np.percentile(sim_data[key]["upstream_cost"], risk_percentile_U)
            safe_U_val = -np.percentile(sim_data[key]["flood_damage"], risk_percentile_U)
            env_val    =  np.percentile(sim_data[key]["ecosystem"], risk_percentile_U)
            
            econ_D_val = -np.percentile(sim_data[key]["downstream_cost"], risk_percentile_D)
            safe_D_val = -np.percentile(sim_data[key]["flood_damage"], risk_percentile_D)
            env_D_val  =  np.percentile(sim_data[key]["ecosystem"], risk_percentile_D)
            
            raw_payoffs_U[key] = {"econ": econ_U_val, "safe": safe_U_val, "env": env_val}
            raw_payoffs_D[key] = {"econ": econ_D_val, "safe": safe_D_val, "env": env_D_val}

# ================================================================
# (4) Normalization of each component (to 0–1 scale)
#     For costs and damage (econ, safe): lower is better → use 1 – (v-min)/(max-min)
#     For ecosystem (env): higher is better → use (v-min)/(max-min)
# ================================================================
def normalize_cost(v, v_min, v_max):
    if v_max == v_min:
        return 0.5
    return (v - v_min) / (v_max - v_min)

def normalize_env(v, v_min, v_max):
    if v_max == v_min:
        return 0.5
    return (v - v_min) / (v_max - v_min)

def get_min_max(component, raw_payoffs):
    values = [raw_payoffs[(u,d)][component] for u in strategies_U for d in strategies_D]
    return min(values), max(values)

econ_U_min, econ_U_max = get_min_max("econ", raw_payoffs_U)
safe_U_min, safe_U_max = get_min_max("safe", raw_payoffs_U)
env_U_min,  env_U_max  = get_min_max("env", raw_payoffs_U)

econ_D_min, econ_D_max = get_min_max("econ", raw_payoffs_D)
safe_D_min, safe_D_max = get_min_max("safe", raw_payoffs_D)
env_D_min,  env_D_max  = get_min_max("env", raw_payoffs_D)

# ================================================================
# (5) Set weight parameters for each component (modifiable)
# ================================================================
# Upstream weights for econ_U, safe_U, env_U
wU_econ = 1.0
wU_safe = 0.5
wU_env  = 1.0 

# Downstream weights for econ_D, safe_D, env_D
wD_econ = 1.0
wD_safe = 1.0
wD_env  = 0.5

# ================================================================
# (6) Calculate final composite payoffs from normalized components.
#     For upstream, we define:
#       final_payoff_U = (wU_econ * norm_econ_U + wU_safe * norm_safe_U + wU_env * norm_env_U)
#                      / (wU_econ + wU_safe + wU_env)
#     Similarly for downstream.
# ================================================================
final_payoffs_U = {}
final_payoffs_D = {}

for u in strategies_U:
    for d in strategies_D:
        key = (u, d)
        if key in raw_payoffs_U:
            norm_econ_U = normalize_cost(raw_payoffs_U[key]["econ"], econ_U_min, econ_U_max)
            norm_safe_U = normalize_cost(raw_payoffs_U[key]["safe"], safe_U_min, safe_U_max)
            norm_env_U  = normalize_env(raw_payoffs_U[key]["env"], env_U_min, env_U_max)
            final_payoffs_U[key] = (wU_econ * norm_econ_U + wU_safe * norm_safe_U + wU_env * norm_env_U) / (wU_econ + wU_safe + wU_env)

            norm_econ_D = normalize_cost(raw_payoffs_D[key]["econ"], econ_D_min, econ_D_max)
            norm_safe_D = normalize_cost(raw_payoffs_D[key]["safe"], safe_D_min, safe_D_max)
            norm_env_D  = normalize_env(raw_payoffs_D[key]["env"], env_D_min, env_D_max)
            final_payoffs_D[key] = (wD_econ * norm_econ_D + wD_safe * norm_safe_D + wD_env * norm_env_D) / (wD_econ + wD_safe + wD_env)

# ================================================================
# (7) Game theory analysis functions (using the computed composite payoffs)
# ================================================================
def payoff_U(sU, sD):
    return final_payoffs_U[(sU, sD)]

def payoff_D(sU, sD):
    return final_payoffs_D[(sU, sD)]

# (A) Find pure-strategy Nash equilibria
def find_nash_equilibria():
    nash_list = []
    for sU in strategies_U:
        for sD in strategies_D:
            u_val = payoff_U(sU, sD)
            u_best = all(payoff_U(altU, sD) <= u_val + 1e-10 for altU in strategies_U)
            d_val = payoff_D(sU, sD)
            d_best = all(payoff_D(sU, altD) <= d_val + 1e-10 for altD in strategies_D)
            if u_best and d_best:
                nash_list.append((sU, sD))
    return nash_list

# (B) Cooperative game analysis: characteristic function and core, Shapley value.
def coalition_value(S):
    if len(S) == 0:
        return 0.0
    elif S == {"U"}:
        best_val = max(payoff_U(sU, "No") for sU in strategies_U)  # D baseline = "No"
        return best_val
    elif S == {"D"}:
        best_val = max(payoff_D("No", sD) for sD in strategies_D)  # U baseline = "No"
        return best_val
    elif S == {"U", "D"}:
        best_val = max(payoff_U(sU, sD) + payoff_D(sU, sD) for sU in strategies_U for sD in strategies_D)
        return best_val

def check_core_2players():
    vU = coalition_value({"U"})
    vD = coalition_value({"D"})
    vUD = coalition_value({"U", "D"})
    if vUD >= vU + vD - 1e-6:
        return True, vU, vD, vUD
    else:
        return False, vU, vD, vUD

def shapley_value_2players():
    v_empty = coalition_value(set())
    vU = coalition_value({"U"})
    vD = coalition_value({"D"})
    vUD = coalition_value({"U", "D"})
    phi_U = 0.5 * (vU - v_empty) + 0.5 * (vUD - vD)
    phi_D = 0.5 * (vD - v_empty) + 0.5 * (vUD - vU)
    return (phi_U, phi_D)

# (C) Borda rule demonstration
def social_payoff(sU, sD):
    return payoff_U(sU, sD) + payoff_D(sU, sD)

def borda_winner():
    scenarios = [(u, d) for u in strategies_U for d in strategies_D]
    sorted_by_U = sorted(scenarios, key=lambda sc: payoff_U(sc[0], sc[1]), reverse=True)
    rank_U = {sc: i for i, sc in enumerate(sorted_by_U)}
    sorted_by_D = sorted(scenarios, key=lambda sc: payoff_D(sc[0], sc[1]), reverse=True)
    rank_D = {sc: i for i, sc in enumerate(sorted_by_D)}
    borda_scores = {sc: rank_U[sc] + rank_D[sc] for sc in scenarios}
    winner = min(borda_scores, key=borda_scores.get)
    return winner, borda_scores

# ================================================================
# (8) Result output and visualization
# ================================================================
if __name__ == "__main__":
    print("==== Non-cooperative Analysis (Nash Equilibria) ====")
    NE = find_nash_equilibria()
    if NE:
        for (sU, sD) in NE:
            print(f"  Equilibrium: (U = {sU}, D = {sD}) -> (U payoff = {payoff_U(sU, sD):.3f}, D payoff = {payoff_D(sU, sD):.3f})")
    else:
        print("  No pure-strategy Nash Equilibrium found.")
    
    print("\n==== Cooperative Game Analysis ====")
    has_core, vU_val, vD_val, vUD_val = check_core_2players()
    print(f"  v(U)   = {vU_val:.3f}")
    print(f"  v(D)   = {vD_val:.3f}")
    print(f"  v(U,D) = {vUD_val:.3f}")
    if has_core:
        print("  Core is non-empty (v(U,D) >= v(U) + v(D)).")
    else:
        print("  Core is empty (v(U,D) < v(U) + v(D)).")
    phi_U, phi_D = shapley_value_2players()
    print(f"  Shapley Value: phi_U = {phi_U:.3f}, phi_D = {phi_D:.3f}")
    
    print("\n==== Borda Rule ====")
    winner, scores = borda_winner()
    for sc in sorted(scores, key=scores.get):
        print(f"  Scenario {sc}: Score = {scores[sc]} (U payoff = {payoff_U(sc[0], sc[1]):.3f}, D payoff = {payoff_D(sc[0], sc[1]):.3f})")
    print(f"  Borda winner: {winner} (U payoff = {payoff_U(winner[0], winner[1]):.3f}, D payoff = {payoff_D(winner[0], winner[1]):.3f}, total = {social_payoff(winner[0], winner[1]):.3f})")
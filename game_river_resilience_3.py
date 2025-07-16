# 上記のコードの上部は同じ
import pandas as pd
import numpy as np
import itertools

csv_path = 'summary_results.csv'

# プレイヤー分割
upstream_strategies = list(itertools.product([0, 100], [0.0, 10.0]))
downstream_strategies = list(itertools.product([0, 2.0], [0.0, 100]))

df = pd.read_csv(csv_path)

# Gen3のみ使用
df = df[['RCP', 'Decision_ID',
         'Flood Damage_Gen3', 'Ecosystem Level_Gen3', 'Municipal Cost_Gen3',
         'planting_trees_amount', 'dam_levee_construction_cost',
         'house_migration_amount', 
         'paddy_dam_construction_cost']]
        #  'capacity_building_cost']]

# df['Upstream_Cost'] = df['Municipal Cost_Gen3'] * (
#     (df['planting_trees_amount'] + df['dam_levee_construction_cost']*50) /
#     (df['planting_trees_amount'] + df['dam_levee_construction_cost']*50 +
#      df['house_migration_amount'] + df['capacity_building_cost']*50 + 1e-6)
# )

df['Upstream_Cost'] = df['planting_trees_amount'] * 2310000 + df['paddy_dam_construction_cost'] * 1000000
df['Downstream_Cost'] = df['Municipal Cost_Gen3'] - df['Upstream_Cost']

# =======================
# 汎用関数
# =======================
def normalize(val, min_val, max_val, reverse=False):
    if max_val == min_val:
        return 0.5
    norm = (val - min_val) / (max_val - min_val)
    return 1 - norm if reverse else norm

def normalize_all(raw_dict):
    keys = list(raw_dict.keys())
    all_vals = {'cost': [], 'flood': [], 'eco': []}
    for key in keys:
        for k in all_vals:
            all_vals[k].append(raw_dict[key][k])
    min_max = {k: (min(all_vals[k]), max(all_vals[k])) for k in all_vals}

    norm_dict = {}
    for key in keys:
        norm_dict[key] = {
            'cost': normalize(raw_dict[key]['cost'], *min_max['cost'], reverse=False),
            'flood': normalize(raw_dict[key]['flood'], *min_max['flood'], reverse=False),
            'eco': normalize(raw_dict[key]['eco'], *min_max['eco'], reverse=False)
        }
    return norm_dict

def weighted_score(p, w):
    return sum(p[k] * w[k] for k in w) / sum(w.values())

# =======================
# パラメータを指定してナッシュ均衡を計算
# =======================
def compute_equilibria(eco_weight_U, eco_weight_D, rcp_weight_U_19, rcp_weight_D_19):
    rcp_weights_U = {'RCP1.9': rcp_weight_U_19, 'RCP8.5': 1 - rcp_weight_U_19}
    rcp_weights_D = {'RCP1.9': rcp_weight_D_19, 'RCP8.5': 1 - rcp_weight_D_19}

    payoffs_U = {}
    payoffs_D = {}

    for u_strat in upstream_strategies:
        for d_strat in downstream_strategies:
            mask = (
                # (df['planting_trees_amount'] == u_strat[0]) &
                # (df['dam_levee_construction_cost'] == u_strat[1]) &
                # (df['house_migration_amount'] == d_strat[0]) &
                # (df['capacity_building_cost'] == d_strat[1])
                (df['planting_trees_amount'] == u_strat[0]) &
                (df['paddy_dam_construction_cost'] == u_strat[1]) &
                (df['dam_levee_construction_cost'] == d_strat[0]) &
                (df['house_migration_amount'] == d_strat[1])
            )
            subset = df[mask]
            if subset.empty:
                continue

            flood = 0
            eco = 0
            up_cost = 0
            down_cost = 0

            for rcp in ['RCP1.9', 'RCP8.5']:
                weight_U = rcp_weights_U[rcp]
                weight_D = rcp_weights_D[rcp]
                rcp_df = subset[subset['RCP'] == rcp]
                if not rcp_df.empty:
                    flood += 0.5 * (weight_U + weight_D) * rcp_df['Flood Damage_Gen3'].mean()
                    eco += 0.5 * (weight_U + weight_D) * rcp_df['Ecosystem Level_Gen3'].mean()
                    up_cost += weight_U * rcp_df['Upstream_Cost'].mean()
                    down_cost += weight_D * rcp_df['Downstream_Cost'].mean()

            payoffs_U[(u_strat, d_strat)] = {'cost': -up_cost, 'flood': -flood, 'eco': eco}
            payoffs_D[(u_strat, d_strat)] = {'cost': -down_cost, 'flood': -flood, 'eco': eco}

    payoffs_U_norm = normalize_all(payoffs_U)
    payoffs_D_norm = normalize_all(payoffs_D)

    weights_U = {'cost': 1.0, 'flood': 1.0, 'eco': eco_weight_U}
    weights_D = {'cost': 1.0, 'flood': 1.0, 'eco': eco_weight_D}

    final_U = {k: weighted_score(payoffs_U_norm[k], weights_U) for k in payoffs_U_norm}
    final_D = {k: weighted_score(payoffs_D_norm[k], weights_D) for k in payoffs_D_norm}

    nash_equilibria = []
    for u in upstream_strategies:
        for d in downstream_strategies:
            if (u, d) not in final_U:
                continue
            u_val = final_U[(u, d)]
            d_val = final_D[(u, d)]
            u_best = all(final_U.get((alt_u, d), -np.inf) <= u_val + 1e-6 for alt_u in upstream_strategies)
            d_best = all(final_D.get((u, alt_d), -np.inf) <= d_val + 1e-6 for alt_d in downstream_strategies)
            if u_best and d_best:
                nash_equilibria.append(((u, d), u_val, d_val))

    return nash_equilibria

# =======================
# eco重み or RCP重みを変化させた場合の比較
# =======================
def analyze_equilibria_over_params(param_type='eco'):
    results = []
    for v in np.linspace(0, 1, 11):
        if param_type == 'eco':
            eco_weight_U = v
            eco_weight_D = 0.0  # 固定
            rcp_weight_U_19 = 0.5
            rcp_weight_D_19 = 0.5
        else:
            eco_weight_U = 0.5
            eco_weight_D = 0.0
            rcp_weight_U_19 = 0.5
            rcp_weight_D_19 = v  # または 0.5 固定でも可

        equilibria = compute_equilibria(
            eco_weight_U, eco_weight_D, rcp_weight_U_19, rcp_weight_D_19
        )
        results.append({'param': v, 'nash_count': len(equilibria), 'equilibria': equilibria})
    return results


def compute_payoff_matrices(U_types, D_types, weights_U, weights_D):
    """
    各タイプの組み合わせごとの利得行列を計算
    """
    payoff_U = {}
    payoff_D = {}

    for i, rcp_U in enumerate(U_types):
        for j, rcp_D in enumerate(D_types):
            key = (i, j)
            payoff_U[key] = {}
            payoff_D[key] = {}

            for u_strat in upstream_strategies:
                for d_strat in downstream_strategies:
                    mask = (
                        # (df['planting_trees_amount'] == u_strat[0]) &
                        # (df['dam_levee_construction_cost'] == u_strat[1]) &
                        # (df['house_migration_amount'] == d_strat[0]) &
                        # (df['capacity_building_cost'] == d_strat[1])
                        (df['planting_trees_amount'] == u_strat[0]) &
                        (df['paddy_dam_construction_cost'] == u_strat[1]) &
                        (df['dam_levee_construction_cost'] == d_strat[0]) &
                        (df['house_migration_amount'] == d_strat[1])
                    )
                    subset = df[mask]
                    if subset.empty:
                        continue

                    flood = 0
                    eco = 0
                    up_cost = 0
                    down_cost = 0

                    for rcp in ['RCP1.9', 'RCP8.5']:
                        wU = rcp_U[rcp]
                        wD = rcp_D[rcp]
                        rcp_df = subset[subset['RCP'] == rcp]
                        if not rcp_df.empty:
                            flood += 0.5 * (wU + wD) * rcp_df['Flood Damage_Gen3'].mean()
                            eco += 0.5 * (wU + wD) * rcp_df['Ecosystem Level_Gen3'].mean()
                            up_cost += wU * rcp_df['Upstream_Cost'].mean()
                            down_cost += wD * rcp_df['Downstream_Cost'].mean()

                    # ペイオフの格納
                    payoff_U[key][(u_strat, d_strat)] = {
                        'cost': -up_cost, 'flood': -flood, 'eco': eco
                    }
                    payoff_D[key][(u_strat, d_strat)] = {
                        'cost': -down_cost, 'flood': -flood, 'eco': eco
                    }

    # 正規化
    for key in payoff_U:
        payoff_U[key] = normalize_all(payoff_U[key])
        payoff_D[key] = normalize_all(payoff_D[key])

    return payoff_U, payoff_D


def find_bayesian_nash_equilibrium(U_types, D_types, U_belief_on_D, D_belief_on_U, weights_U, weights_D):
    """
    各プレイヤーのタイプごとに期待利得を計算し、BNEを求める
    """
    payoff_U, payoff_D = compute_payoff_matrices(U_types, D_types, weights_U, weights_D)

    # 各タイプごとの最適戦略
    best_strategies_U = {}
    best_strategies_D = {}

    # Upstreamプレイヤーの最適戦略
    for i in range(len(U_types)):
        best_val = -np.inf
        best_strat = None
        for u in upstream_strategies:
            expected = 0
            for j in range(len(D_types)):
                belief = U_belief_on_D[j]
                for d in downstream_strategies:
                    key = (i, j)
                    if (u, d) in payoff_U[key]:
                        score = weighted_score(payoff_U[key][(u, d)], weights_U)
                        expected += belief * score
            if expected > best_val:
                best_val = expected
                best_strat = u
        best_strategies_U[f"U_type_{i+1}"] = best_strat

    # Downstreamプレイヤーの最適戦略
    for j in range(len(D_types)):
        best_val = -np.inf
        best_strat = None
        for d in downstream_strategies:
            expected = 0
            for i in range(len(U_types)):
                belief = D_belief_on_U[i]
                for u in upstream_strategies:
                    key = (i, j)
                    if (u, d) in payoff_D[key]:
                        score = weighted_score(payoff_D[key][(u, d)], weights_D)
                        expected += belief * score
            if expected > best_val:
                best_val = expected
                best_strat = d
        best_strategies_D[f"D_type_{j+1}"] = best_strat

    return best_strategies_U, best_strategies_D


# =======================
# 実行例（NE）
# =======================
eco_results = analyze_equilibria_over_params('eco')
rcp_results = analyze_equilibria_over_params('rcp')

# 結果の一部を表示
for res in eco_results:
    print(f"[Up_eco={res['param']:.1f}] Nash数: {res['nash_count']}")
    print(res['equilibria'])

for res in rcp_results:
    print(f"[rcp_D19={res['param']:.1f}] Nash数: {res['nash_count']}")
    print(res['equilibria'])

# # =======================
# # 実行例（BNE）
# # =======================
# # タイプ定義
# U_types = [{'RCP1.9': 0.9, 'RCP8.5': 0.1}, {'RCP1.9': 0.1, 'RCP8.5': 0.9}]
# D_types = [{'RCP1.9': 0.9, 'RCP8.5': 0.1}, {'RCP1.9': 0.1, 'RCP8.5': 0.9}]

# # 信念（ここは変化させる予定）
# U_belief_on_D = [0.7, 0.3]
# D_belief_on_U = [0.4, 0.6]

# # 重み
# weights_U = {'cost': 1.0, 'flood': 1.0, 'eco': 0.5}
# weights_D = {'cost': 1.0, 'flood': 1.0, 'eco': 0.0}

# # BNE実行
# U_result, D_result = find_bayesian_nash_equilibrium(
#     U_types, D_types, U_belief_on_D, D_belief_on_U, weights_U, weights_D
# )

# print("=== Bayesian Nash Equilibrium ===")
# for k, v in U_result.items():
#     print(f"{k} chooses strategy {v}")
# for k, v in D_result.items():
#     print(f"{k} chooses strategy {v}")

# # ----------------------------------------

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import itertools

# # 仮の戦略（例として一部）
# upstream_strategies = list(itertools.product([0, 50], [0.0, 1.0]))
# downstream_strategies = list(itertools.product([0, 50], [0.0, 5.0]))

# # 仮の関数：ベイズナッシュ均衡結果の戦略選択インデックスを返す（例として）
# def simulate_bne(belief_val):
#     """
#     belief_val: プレイヤーUがDのタイプ1を信じる確率（Dのタイプ2は 1 - belief_val）
#     """
#     # 簡略化された仮の結果
#     return {
#         'U_type_1': upstream_strategies[int(belief_val < 0.5)],
#         'U_type_2': upstream_strategies[int(belief_val >= 0.5)],
#         'D_type_1': downstream_strategies[int(belief_val > 0.5)],
#         'D_type_2': downstream_strategies[int(belief_val <= 0.5)],
#     }

# # スイープ
# belief_values = np.linspace(0, 1, 11)
# strategy_indices = []

# for b in belief_values:
#     result = simulate_bne(b)
#     strategy_indices.append({
#         'belief': b,
#         'U1': upstream_strategies.index(result['U_type_1']),
#         'U2': upstream_strategies.index(result['U_type_2']),
#         'D1': downstream_strategies.index(result['D_type_1']),
#         'D2': downstream_strategies.index(result['D_type_2']),
#     })

# df_result = pd.DataFrame(strategy_indices)
# print(df_result)

# # プロット
# plt.figure()
# plt.plot(df_result['belief'], df_result['U1'], label='U_type_1 Strategy Index')
# plt.plot(df_result['belief'], df_result['U2'], label='U_type_2 Strategy Index')
# plt.plot(df_result['belief'], df_result['D1'], label='D_type_1 Strategy Index')
# plt.plot(df_result['belief'], df_result['D2'], label='D_type_2 Strategy Index')
# plt.xlabel('Belief that D is Type 1 (P(D1))')
# plt.ylabel('Strategy Index')
# plt.title('Bayesian Nash Equilibrium vs. Belief')
# plt.legend()
# plt.grid(True)
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import csv
import json

# ===============================
# シミュレーションの基本パラメータ
# ===============================
START_YEAR = 2025
END_YEAR   = 2100
YEARS = np.arange(START_YEAR, END_YEAR + 1)

INITIAL_FOREST_CAPACITY    = 100.0
INITIAL_DAM_CAPACITY       = 20.0
INITIAL_EMBANKMENT_CAPACITY = 40.0
INITIAL_HOUSES             = 5000
INITIAL_ECOSYSTEM          = 100.0

FOREST_CAPACITY_INCREMENT     = 10.0
DAM_CAPACITY_INCREMENT        = 10.0
EMBANKMENT_CAPACITY_INCREMENT = 10.0

DELAY_FOREST     = 40
DELAY_DAM        = 10
DELAY_EMBANKMENT = 5

COST_FOREST_PER_UNIT      = 200000
COST_DAM_PER_UNIT         = 50000000
COST_EMBANKMENT_PER_UNIT  = 100000000
COST_RELOCATION_PER_HOUSE = 30000000

FOREST_ECO_FACTOR     = 0.2
DAM_ECO_FACTOR        = 0.1
EMBANKMENT_ECO_FACTOR = 0.05

DAMAGE_COEFFICIENT = 10000

def generate_rainfall(year):
    base_mean = 300 + (year - START_YEAR)
    std = 50
    R = np.random.normal(base_mean, std)
    extreme_prob = 0.10 + (year - START_YEAR)*0.001
    if np.random.rand() < extreme_prob:
        R += np.random.uniform(100, 200)
    return max(R, 0.0)

def simulate_run(policy_actions):
    forest_capacity     = INITIAL_FOREST_CAPACITY
    dam_capacity        = INITIAL_DAM_CAPACITY
    embankment_capacity = INITIAL_EMBANKMENT_CAPACITY
    houses_at_risk      = INITIAL_HOUSES
    ecosystem           = INITIAL_ECOSYSTEM

    pending_actions = []
    upstream_cost   = 0.0
    downstream_cost = 0.0
    cumulative_damage = 0.0

    policy_by_year = {}
    for act in policy_actions:
        policy_by_year.setdefault(act["year"], []).append(act)

    for year in YEARS:
        new_pending = []
        for (eff_year, act_type, amount) in pending_actions:
            if year >= eff_year:
                if act_type == "forest":
                    forest_capacity += FOREST_CAPACITY_INCREMENT * amount
                elif act_type == "dam":
                    multiplier = np.random.uniform(0.8, 1.2)
                    dam_capacity += DAM_CAPACITY_INCREMENT * amount * multiplier
                elif act_type == "embankment":
                    multiplier = np.random.uniform(0.8, 1.2)
                    embankment_capacity += EMBANKMENT_CAPACITY_INCREMENT * amount * multiplier
            else:
                new_pending.append((eff_year, act_type, amount))
        pending_actions = new_pending

        if year in policy_by_year:
            for act in policy_by_year[year]:
                if act["type"] == "forest":
                    pending_actions.append((year + DELAY_FOREST, "forest", act["amount"]))
                    cost = COST_FOREST_PER_UNIT * FOREST_CAPACITY_INCREMENT * act["amount"]
                    upstream_cost += cost
                elif act["type"] == "dam":
                    delay_offset = np.random.randint(-1, 2)
                    pending_actions.append((year + DELAY_DAM + delay_offset, "dam", act["amount"]))
                    cost = COST_DAM_PER_UNIT * DAM_CAPACITY_INCREMENT * act["amount"]
                    upstream_cost += cost
                elif act["type"] == "embankment":
                    delay_offset = np.random.randint(-1, 2)
                    pending_actions.append((year + DELAY_EMBANKMENT + delay_offset, "embankment", act["amount"]))
                    cost = COST_EMBANKMENT_PER_UNIT * EMBANKMENT_CAPACITY_INCREMENT * act["amount"]
                    downstream_cost += cost
                elif act["type"] == "relocate":
                    houses = act["houses"]
                    houses_at_risk = max(0, houses_at_risk - houses)
                    cost = COST_RELOCATION_PER_HOUSE * houses
                    downstream_cost += cost

        R = generate_rainfall(year)
        threshold = forest_capacity + dam_capacity + embankment_capacity
        if R > threshold:
            damage = (R - threshold) * DAMAGE_COEFFICIENT * houses_at_risk
        else:
            damage = 0.0
        cumulative_damage += damage

        forest_effect     = (forest_capacity - INITIAL_FOREST_CAPACITY) * FOREST_ECO_FACTOR
        dam_effect        = dam_capacity * DAM_ECO_FACTOR
        embankment_effect = (embankment_capacity - INITIAL_EMBANKMENT_CAPACITY) * EMBANKMENT_ECO_FACTOR
        ecosystem = INITIAL_ECOSYSTEM + forest_effect - dam_effect - embankment_effect + np.random.normal(0, 2)

    results = {
        "cumulative_damage": cumulative_damage,
        "upstream_cost": upstream_cost,
        "downstream_cost": downstream_cost,
        "final_ecosystem": ecosystem
    }
    return results

def monte_carlo_simulation(n_runs, policy_actions, seed=None):
    if seed is not None:
        np.random.seed(seed)
    results_list = []
    for _ in range(n_runs):
        results_list.append(simulate_run(policy_actions))
    return results_list

def compute_percentiles(data):
    pct = np.percentile(data, [1, 5, 25, 50, 75, 95, 99])
    return {p: v for p, v in zip([1, 5, 25, 50, 75, 95, 99], pct)}

def generate_policy_actions(forest_decision, dam_decision, embankment_decision, relocation_decision):
    actions = []
    if forest_decision:
        actions.append({"type": "forest", "year": 2025, "amount": 1})
    if dam_decision:
        actions.append({"type": "dam", "year": 2030, "amount": 1})
    if embankment_decision:
        actions.append({"type": "embankment", "year": 2035, "amount": 1})
    if relocation_decision:
        actions.append({"type": "relocate", "year": 2040, "houses": 100})
    return actions

# シナリオ名を分かりやすくまとめるための関数
def scenario_label(forest, dam, embankment, relocation):
    """
    例: F- D+ E- R+ のように短く表現する
    """
    f_str = "F+" if forest else "F-"
    d_str = "D+" if dam else "D-"
    e_str = "E+" if embankment else "E-"
    r_str = "R+" if relocation else "R-"
    return f"{f_str}/{d_str}/{e_str}/{r_str}"

if __name__ == "__main__":
    n_runs = 1000
    decisions = [False, True]

    sim_data_out = {}
    results_matrix = {}
    scenario_data = {}
    scatter_points = []

    # 全16通りの組合せをループ
    for forest in decisions:
        for dam in decisions:
            for embankment in decisions:
                for relocation in decisions:
                    # シナリオ名を短く分かりやすく（例: 'F+/D-/E+/R-' など）
                    label_text = scenario_label(forest, dam, embankment, relocation)

                    policy_actions = generate_policy_actions(forest, dam, embankment, relocation)
                    sim_results = monte_carlo_simulation(n_runs, policy_actions, seed=42)

                    upstream_costs   = np.array([res["upstream_cost"] for res in sim_results])
                    ecosystems       = np.array([res["final_ecosystem"] for res in sim_results])
                    flood_damages    = np.array([res["cumulative_damage"] for res in sim_results])
                    downstream_costs = np.array([res["downstream_cost"] for res in sim_results])

                    sim_data_out[(forest, dam, embankment, relocation)] = {
                        "upstream_cost": upstream_costs,
                        "downstream_cost": downstream_costs,
                        "flood_damage": flood_damages,
                        "ecosystem": ecosystems
                    }

                    up_cost_pct   = compute_percentiles(upstream_costs)
                    eco_pct       = compute_percentiles(ecosystems)
                    flood_pct     = compute_percentiles(flood_damages)
                    down_cost_pct = compute_percentiles(downstream_costs)

                    combo_key = label_text
                    results_matrix[combo_key] = {
                        "upstream_cost": up_cost_pct,
                        "ecosystem": eco_pct,
                        "flood_damage": flood_pct,
                        "downstream_cost": down_cost_pct
                    }

                    median_flood = np.median(flood_damages)
                    median_eco   = np.median(ecosystems)
                    scatter_points.append((median_flood, median_eco, combo_key))

                    scenario_data[label_text] = {
                        "flood": flood_damages,
                        "eco": ecosystems,
                        "median_flood": median_flood,
                        "median_eco": median_eco
                    }

    # CSVに書き出す部分は省略または同様に実装可能

    # 可視化: 上記の label_text を使い，凡例の文字サイズを大きく設定
    plt.figure(figsize=(10, 6))
    colors = plt.cm.tab20(np.linspace(0, 1, len(scenario_data)))
    for i, (label_text, data) in enumerate(scenario_data.items()):
        flood_data = data["flood"]
        eco_data   = data["eco"]
        median_flood = data["median_flood"]
        median_eco   = data["median_eco"]

        # 散布点をやや小さく, alpha=0.05 で重なりを緩和
        plt.scatter(flood_data, eco_data, alpha=0.05, s=10, color=colors[i])
        # 中央値を明確に表示
        plt.scatter(median_flood, median_eco, marker='D', s=120, edgecolor='k',
                    color=colors[i], label=label_text)

    plt.xlabel("Flood Damage", fontsize=16)
    plt.ylabel("Ecosystem Index", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # plt.title("Scenario Comparison: Flood Damage vs. Ecosystem Index")
    plt.grid(True)
    # 凡例のフォントサイズを 'medium' や 'large' などで調整
    plt.legend(fontsize='large', bbox_to_anchor=(1, 1), loc='upper right')
    plt.tight_layout()
    plt.show()

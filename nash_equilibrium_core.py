#!/usr/bin/env python3

# ================================================================
# Example: Upstream (U) vs. Downstream (D) with payoff depending
# on (environment, economy, safety) plus weights.

#  1) Non-cooperative analysis (Nash Equilibria)
#  2) Cooperative analysis: core existence check, Shapley Value
#  3) A simple Borda rule demonstration
# ================================================================

import itertools

# 1. Strategy labels
strategies_U = ["NP", "PT", "IL"]  # Upstream: No-Plant, Plant-Trees, Invest-Levee
strategies_D = ["DN", "IL", "RE"]  # Downstream: Do-Nothing, Invest-Levee, Relocation

# 2. Example (env, econ, safe) payoff dictionaries for each (sU, sD) from U and D perspective

env_U = {
    ("NP","DN"): -1, ("NP","IL"): -2, ("NP","RE"): -1,
    ("PT","DN"): +2, ("PT","IL"): +1, ("PT","RE"): +2,
    ("IL","DN"): -2, ("IL","IL"): -3, ("IL","RE"): -2
}
econ_U = {
    ("NP","DN"): +2, ("NP","IL"): +1, ("NP","RE"): +2,
    ("PT","DN"): -1, ("PT","IL"): -2, ("PT","RE"): -1,
    ("IL","DN"): -1, ("IL","IL"): -2, ("IL","RE"): -1
}
safe_U = {
    ("NP","DN"):  0, ("NP","IL"): +1, ("NP","RE"):  0,
    ("PT","DN"): +1, ("PT","IL"): +2, ("PT","RE"): +1,
    ("IL","DN"): +2, ("IL","IL"): +3, ("IL","RE"): +2
}

env_D = {
    ("NP","DN"):  0, ("NP","IL"): -1, ("NP","RE"):  0,
    ("PT","DN"): +1, ("PT","IL"):  0, ("PT","RE"): +1,
    ("IL","DN"): -1, ("IL","IL"): -2, ("IL","RE"): -1
}
econ_D = {
    ("NP","DN"):  0, ("NP","IL"): -1, ("NP","RE"): -2,
    ("PT","DN"):  0, ("PT","IL"): -1, ("PT","RE"): -2,
    ("IL","DN"): -1, ("IL","IL"): -2, ("IL","RE"): -3
}
safe_D = {
    ("NP","DN"): -1, ("NP","IL"): +2, ("NP","RE"): +3,
    ("PT","DN"):  0, ("PT","IL"): +2, ("PT","RE"): +3,
    ("IL","DN"): +1, ("IL","IL"): +3, ("IL","RE"): +4
}

# 3. Set weights for both players (example)
wU_env, wU_econ, wU_safe = 1, 1, 1   # Upstream weights
wD_env, wD_econ, wD_safe = 1, 1, 1   # Downstream weights

def payoff_U(sU, sD):
    return (wU_env * env_U[(sU,sD)]
          + wU_econ * econ_U[(sU,sD)]
          + wU_safe * safe_U[(sU,sD)])

def payoff_D(sU, sD):
    return (wD_env * env_D[(sU,sD)]
          + wD_econ * econ_D[(sU,sD)]
          + wD_safe * safe_D[(sU,sD)])

# 4. Find pure-strategy Nash equilibria (non-cooperative)
def find_nash_equilibria():
    nash_list = []
    for sU in strategies_U:
        for sD in strategies_D:
            # Check if sU is best response to sD
            u_val = payoff_U(sU, sD)
            u_best = True
            for altU in strategies_U:
                if payoff_U(altU, sD) > u_val:
                    u_best = False
                    break
            
            # Check if sD is best response to sU
            d_val = payoff_D(sU, sD)
            d_best = True
            for altD in strategies_D:
                if payoff_D(sU, altD) > d_val:
                    d_best = False
                    break
            
            if u_best and d_best:
                nash_list.append((sU, sD))
    return nash_list

# ====================================================================
# 5. Cooperative analysis
#    Define a characteristic function v(S). For a 2-player game:
#    S can be {}, {U}, {D}, or {U,D}.
# ====================================================================
def coalition_value(S):
    """
    Return v(S): the characteristic function for coalition S.
    We'll assume we interpret the 'value' as the sum of payoffs
    to members of S, *under the best strategy combination that S can enforce*.

    For simplicity:
     - If S = {U}, we let U choose sU in strategies_U to maximize U's payoff,
       assuming D does a "baseline" action or "DN" (Do-Nothing).
     - If S = {D}, similarly let D choose sD in strategies_D with U=DN as baseline.
     - If S = {U,D}, then we choose (sU, sD) to maximize payoff_U + payoff_D.
     - If S = {}: v({}) = 0
    """
    if len(S) == 0:
        return 0.0
    elif S == {"U"}:
        # U alone: we treat D as "DN" (downstream does nothing).
        # Then we maximize U's payoff over sU in strategies_U.
        best_val = float("-inf")
        for sU in strategies_U:
            val = payoff_U(sU, "DN")  # D=DN baseline
            if val > best_val:
                best_val = val
        return best_val
    elif S == {"D"}:
        # D alone: we treat U as "NP" (Upstream does nothing).
        # Then we maximize D's payoff over sD in strategies_D.
        best_val = float("-inf")
        for sD in strategies_D:
            val = payoff_D("NP", sD)  # U=NP baseline
            if val > best_val:
                best_val = val
        return best_val
    elif S == {"U","D"}:
        # Coalition of both: choose (sU, sD) to maximize sum of payoffs
        best_val = float("-inf")
        for sU in strategies_U:
            for sD in strategies_D:
                total = payoff_U(sU, sD) + payoff_D(sU, sD)
                if total > best_val:
                    best_val = total
        return best_val

def check_core_2players():
    """
    For a 2-player game, the core is nonempty iff
      v({U,D}) >= v({U}) + v({D}).
    If that holds, the set of feasible payoffs (xU, xD) such that
      xU + xD = v({U,D})
      xU >= v({U})
      xD >= v({D})
    is the core.
    """
    vU = coalition_value({"U"})
    vD = coalition_value({"D"})
    vUD = coalition_value({"U","D"})
    sum_vU_vD = vU + vD

    if vUD >= sum_vU_vD:
        return True, vU, vD, vUD
    else:
        return False, vU, vD, vUD

def shapley_value_2players():
    """
    Shapley value for 2 players {U, D} is given by:
      phi_U = 1/2 [v({U}) - v({})] + 1/2 [v({U,D}) - v({D})]
      phi_D = 1/2 [v({D}) - v({})] + 1/2 [v({U,D}) - v({U})]
    Usually v({}) = 0.

    We'll return (phi_U, phi_D).
    """
    v_empty = coalition_value({})
    vU = coalition_value({"U"})
    vD = coalition_value({"D"})
    vUD = coalition_value({"U","D"})

    phi_U = 0.5 * (vU - v_empty) + 0.5 * (vUD - vD)
    phi_D = 0.5 * (vD - v_empty) + 0.5 * (vUD - vU)
    return (phi_U, phi_D)

# ====================================================================
# 6. Borda rule demonstration (very simplified)
# ====================================================================
def social_payoff(sU, sD):
    """Sum of payoffs for (sU, sD)."""
    return payoff_U(sU, sD) + payoff_D(sU, sD)

def borda_winner():
    """
    We interpret each (sU, sD) as a 'scenario'.
    Each player ranks the scenarios from highest to lowest payoff.
    We assign Borda scores: if there are N=|strategies_U|*|strategies_D| scenarios,
    each player's top choice gets 0 points, second choice gets 1, etc.
    Then sum across players, scenario with the smallest sum is the Borda winner.

    (Note: This is just one approach to Borda scoring.)
    """
    scenarios = []
    for sU in strategies_U:
        for sD in strategies_D:
            scenarios.append((sU, sD))

    # Number of scenarios
    N = len(scenarios)

    # Upstream's ranking
    # Sort scenarios by U's payoff descending
    sorted_by_U = sorted(scenarios, key=lambda sc: payoff_U(sc[0], sc[1]), reverse=True)
    # Assign rank_U: top scenario gets rank 0, next gets 1, ...
    rank_U = {}
    for i, sc in enumerate(sorted_by_U):
        rank_U[sc] = i

    # Downstream's ranking
    sorted_by_D = sorted(scenarios, key=lambda sc: payoff_D(sc[0], sc[1]), reverse=True)
    rank_D = {}
    for i, sc in enumerate(sorted_by_D):
        rank_D[sc] = i

    # Borda score = rank_U + rank_D
    borda_scores = {}
    for sc in scenarios:
        borda_scores[sc] = rank_U[sc] + rank_D[sc]

    # Winner = scenario with smallest Borda score
    winner = min(borda_scores, key=borda_scores.get)
    return winner, borda_scores


if __name__ == "__main__":
    # 1) Non-cooperative: find Nash equilibria
    NE = find_nash_equilibria()
    if len(NE) == 0:
        print("No pure-strategy Nash Equilibrium found.\n")
    else:
        print("Pure-strategy Nash Equilibria:")
        for (sU, sD) in NE:
            print(f"  (U={sU}, D={sD}) -> (uU={payoff_U(sU,sD)}, uD={payoff_D(sU,sD)})")
        print("")

    # 2) Cooperative game: characteristic function & core check
    has_core, vU, vD, vUD = check_core_2players()
    print("===== Cooperative Game Analysis =====")
    print(f" v(U)   = {vU}")
    print(f" v(D)   = {vD}")
    print(f" v(U,D) = {vUD}")
    if has_core:
        print(" Core is NON-empty!")
        print(f"  Condition: v(U,D) >= v(U)+v(D) -> {vUD} >= {vU+vD}")
        print("  Any (xU, xD) with xU+xD = v(U,D), xU>=v(U), xD>=v(D) is in the core.\n")
    else:
        print(" Core is EMPTY.")
        print(f"  Condition fails: {vUD} < {vU+vD}\n")

    # 3) Shapley Value for 2 players
    phi_U, phi_D = shapley_value_2players()
    print(f"Shapley Value:  phi_U = {phi_U:.2f},  phi_D = {phi_D:.2f}\n")

    # 4) Borda rule demonstration
    winner, scores = borda_winner()
    print("===== Borda Rule =====")
    print("Borda scores (lower is better):")
    for sc in sorted(scores, key=scores.get):
        print(f"  Scenario {sc}: Score = {scores[sc]} "
              f"(uU={payoff_U(sc[0],sc[1])}, uD={payoff_D(sc[0],sc[1])})")
    print(f"\n Borda winner: {winner} "
          f"(payoffs => uU={payoff_U(winner[0],winner[1])}, "
          f"uD={payoff_D(winner[0],winner[1])}, "
          f"sum={social_payoff(winner[0], winner[1])})")

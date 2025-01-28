# Nash Equilibrium and Cooperative Game Analysis

This repository contains a Python script `nash_equilibrium_core.py` that demonstrates:
1. **Non-cooperative game analysis** (finding pure-strategy Nash equilibria).
2. **Cooperative game analysis** for a 2-player setting (Upstream \(U\) vs. Downstream \(D\)):
   - Calculating a **characteristic function** \(v(S)\) for each coalition \(S\).
   - Checking whether the **core** is empty or non-empty.
   - Computing the **Shapley Value** (2-player version).
3. A demonstration of a simple **Borda rule** approach to select a “best” scenario from all possible strategy pairs.

This game is motivated by a situation where the Upstream player (e.g., a region) and Downstream player (e.g., another region or municipality) choose actions (e.g., `No-Plant`, `Plant-Trees`, `Invest-Levee`, etc.) that yield payoffs for each dimension (environment, economy, safety). The script then aggregates these payoffs (weighted sums) to find how each player’s payoff changes with different strategy combinations.

## Features

- **Nash Equilibrium (Pure Strategy)**:  
  Each player checks if their strategy is a best response to the other’s strategy, and vice versa.
- **Cooperative Game**:
  - A simple **characteristic function** \(v(S)\) for a 2-player game, where each coalition \(S\subseteq \{U, D\}\) can optimize its own strategies while treating the non-members as fixed or baseline.
  - **Core check**: For 2 players, the core is nonempty if \(v(\{U,D\}) \ge v(\{U\}) + v(\{D\})\).
  - **Shapley Value** for 2 players: a direct formula based on marginal contributions.
- **Borda Rule**:
  - Treats each \((s_U, s_D)\) pair as a scenario.
  - Each player ranks scenarios by their own payoff; we assign Borda scores and pick the lowest total score as the “winner.”

## File Contents

- **`nash_equilibrium_core.py`**  
  The main script performing all of the above analyses. It is organized as follows:
  1. **Strategy definitions** for Upstream (\(U\)) and Downstream (\(D\)).
  2. **Payoff tables** (`env_U`, `econ_U`, `safe_U` etc.) defining environment, economic, and safety contributions for each strategy pair, from each player’s perspective.
  3. **Weight variables** to combine environment/economy/safety into a single numeric payoff.
  4. **Functions** to:
     - Calculate `payoff_U(sU, sD)` and `payoff_D(sU, sD)`.
     - Find pure-strategy Nash equilibria (`find_nash_equilibria()`).
     - Define a characteristic function `coalition_value(S)`.
     - Check the core condition (`check_core_2players()`).
     - Compute Shapley Value for 2 players (`shapley_value_2players()`).
     - Run a simple Borda rule to find a “winning” scenario (`borda_winner()`).
  5. **`main` section** that executes these methods and prints results.

## Installation (with Poetry)

1. **Clone the repository** (or copy the script into your project).

2. **Install Poetry** (if you haven’t already). For example:
   ```bash
   pip install poetry
   ```
   or follow the instructions on [Poetry’s official documentation](https://python-poetry.org/docs/).

3. **Create a `pyproject.toml`** (if it doesn’t exist) and add any dependencies. 
   - In most cases, this script only uses the standard library, so no external dependencies are strictly necessary. 
   - If you decide to use additional packages (e.g. `numpy` for vector operations), include them. For instance:
     ```toml
     [tool.poetry]
     name = "nash-equilibrium-core"
     version = "0.1.0"
     description = "A script for analyzing Nash Equilibria and Cooperative Games"
     authors = ["Your Name <you@example.com>"]
     python = "^3.8"

     [tool.poetry.dependencies]
     python = "^3.8"
     # For instance, if you used 'numpy':
     # numpy = "^1.21.0"

     [tool.poetry.scripts]
     nash = "nash_equilibrium_core:main"

     [tool.poetry.dev-dependencies]
     pytest = "^7.0.0"
     ```

4. **Install the project**:
   ```bash
   poetry install
   ```
   This will create a virtual environment and install the specified dependencies.

5. **Activate the virtual environment**:
   ```bash
   poetry shell
   ```
   or run commands via:
   ```bash
   poetry run python nash_equilibrium_core.py
   ```

## Usage

Once you have installed dependencies and activated your Poetry environment, you can run:

```bash
poetry run python nash_equilibrium_core.py
```

You should see console output describing:
- Any **pure-strategy Nash Equilibria** found (if any).
- The values of the characteristic function (`v({U})`, `v({D})`, `v({U,D})`), plus whether the **core** is empty or not.
- The **Shapley Value** for each player.
- The **Borda rule** ranking and final winner scenario.

### Example Output

```mathematica
Pure-strategy Nash Equilibria:
  (U=NP, D=IL) -> (uU=..., uD=...)

===== Cooperative Game Analysis =====
 v(U)   = ...
 v(D)   = ...
 v(U,D) = ...
 Core is NON-empty!
   Condition: v(U,D) >= v(U)+v(D) -> ...

Shapley Value:  phi_U = ...,  phi_D = ...

===== Borda Rule =====
Borda scores (lower is better):
  Scenario (NP,DN): Score = ...
  Scenario (NP,IL): Score = ...
  ...
 Borda winner: (PT,IL) (payoffs => uU=..., uD=..., sum=...)
```

This explains:
- Which **pure strategies** are mutual best responses (Nash equilibria).
- The **cooperative scenario** that maximizes \(u_U + u_D\) and whether that total synergy is enough to keep both players inside the coalition (core existence).
- How the **Shapley Value** might allocate the total surplus among them.
- Which scenario the **Borda method** picks as collectively best, given players’ rank preferences.

## Customizing the Model

- You can **change the weight variables** `(wU_env, wU_econ, wU_safe)` and `(wD_env, wD_econ, wD_safe)` to reflect different priority profiles for the players.  
- You can **modify or expand the payoff dictionaries** (`env_U`, etc.) if you want to test more strategy pairs or incorporate real data (like actual flood mitigation benefits, forest regrowth rates, etc.).
- For a multi-player or multi-scenario game, you might adapt the characteristic function logic to handle more coalitions.

## License

[MIT License](LICENSE)

## Contributing

Please open an issue or pull request if you find a bug or have ideas to expand the game model (e.g. multi-period scenarios, Bayesian/incomplete-information, dynamic programming, etc.).

## Author

Takuya Nakashima
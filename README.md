# Mathematical Modelling for Debt Collection

This notebook provides a statistical mathematical model based on optimisation with Multi Armed Bandit (MAB). The project is structured in phases, evolving from a basic logistical challenge into an advanced, self-learning system that adapts its strategy based on real-time feedback.
It starts on with routing problem on how to route the debt collectors based on time and availibility then optimises their route and finally uses MAB to determine and probalisticly run simulations and track metrics over 1000 runs.

---

## Assumptions
These assumptions were made to model the debtor behaviour. This data can vary and based on dynamic situations it can be changed to model new behaviour and track metrics.
``` python
COLLECTOR_SCHEDULES = {
    'C1': {'start': 7, 'end': 15},   # 7 AM to 3 PM
    'C2': {'start': 13, 'end': 22}   # 1 PM to 10 PM
}

VISIT_TIME_MINS_AVG = 20
VISIT_TIME_MINS_VAR = 8
TRAVEL_SPEED_KMH = 25
MAX_DAILY_TRAVEL_KM = 100
MIN_TIME_BETWEEN_VISITS = 5

COLLECTION_SUCCESS_RATE = 0.3
COLLECTION_AMOUNT_PCT_AVG = 0.6
COLLECTION_AMOUNT_PCT_VAR = 0.2

PERSON_AVAILABLE_PROB = 0.7
OUT_OF_STATION_PROB = 0.05
OUT_OF_STATION_DAYS_AVG = 5
OUT_OF_STATION_DAYS_VAR = 3
```


## Phase 1: The Foundational Routing Problem

The notebook begins by solving the most straightforward challenge in field collection: **The Traveling Salesperson Problem (TSP)**. The goal is to determine the most efficient daily routes for collection agents to visit a list of debtors.

### 1. Simulating the Environment
The system first generates a realistic virtual world to operate in.
* **Debtor Generation**: Instead of being random, debtors are created in geographical **clusters** around specific locations in Delhi (e.g., Connaught Place). Each debtor is assigned coordinates (latitude, longitude) and an amount of `money_owed`. This clustering approach mimics real-world demographics.
* **Employee Generation**: Collection agents are also generated and assigned to a primary cluster, simulating a regional operational model.

### 2. Calculating Travel and Optimizing Routes
A key component is determining the time and distance between locations.
* **Initial Approach (Haversine Distance)**: The first prototype uses the `geodesic` distance, which calculates the shortest path on a sphere ("as the crow flies"). This is a simple but unrealistic measure for city navigation.
* **Advanced Approach (OSRM API)**: For realism, the model is upgraded to use the **Open Source Routing Machine (OSRM)**. This free API provides actual road network distances and estimated travel times, accounting for streets, one-way roads, and other real-world constraints. To manage API usage, the code cleverly uses caching (`@lru_cache`) to store results and avoid redundant calls.
* **Route Optimization**:
    * An initial route is created using the **Nearest Neighbor** algorithm, a simple "greedy" method where the agent always travels to the closest unvisited debtor.
    * This route is then significantly improved by the **2-Opt heuristic**. This algorithm iteratively swaps pairs of route segments to see if any swap results in a shorter total path, repeating until no more improvements can be found.

---

## Phase 2: Introducing Behavioral and Temporal Complexity

This phase acknowledges that debt collection is more than just logistics. It's a human-centric problem where timing and behavior are critical. The model evolves from a static routing puzzle into a dynamic, multi-day simulation.

* **Object-Oriented Modeling**: The system is refactored into classes. `DebtorProfile` objects now contain not just location data, but also behavioral traits like `availability_pattern` (e.g., more likely to be home in the evenings) and `payment_willingness`. `Collector` objects are given fixed work schedules (e.g., 7 AM - 3 PM).
* **Probabilistic Visit Outcomes**: A visit is no longer a guaranteed event. The `simulate_visit_outcome` function introduces uncertainty. The success of a visit depends on a series of probabilistic checks:
    1.  Is the person available at this specific hour? (Based on their `availability_pattern`).
    2.  If available, are they willing to pay? (Based on `COLLECTION_SUCCESS_RATE` and their `payment_willingness`).
    3.  If they pay, how much do they pay? (Based on a percentage of the amount owed).

---

## Phase 3: The Adaptive Learning System (Multi-Armed Bandits)

This is the most advanced and powerful part of the project. The system is equipped with a `SmartSchedulingAlgorithm` that learns from the outcomes of past visits to make increasingly better decisions. It tackles the classic **exploration vs. exploitation** dilemma using a framework known as **Multi-Armed Bandits (MAB)**.

### 1. Modeling Debtor Behavior (The "Ground Truth")
To test the learning algorithm, the simulation first creates a complex, hidden reality for each debtor. The MAB algorithm does **not** know this information beforehand; its entire purpose is to discover these patterns through interaction.

* **Debtor Archetypes**: Debtors are assigned a persona like `office_worker`, `homemaker`, or `shift_worker`. This archetype determines the fundamental patterns of their life.
* **Dynamic Financial State**: The simulation models each debtor's financial health on a daily basis.
    * They receive a `monthly_income` on a specific `salary_date`.
    * They have daily expenses, causing their available funds to decrease over the month.
    * This leads to dynamic states: a debtor might be **`FRESH`** right after being paid, but become **`DEPLETED`** (low on funds) towards the end of the month. A visit during a `DEPLETED` state is highly unlikely to yield a payment.
* **Psychological Modeling**: The model also simulates a simple psychological state.
    * Repeated visits increase a **`pressure_level`**.
    * This pressure can make a debtor in a **`PRESSURED`** state more likely to pay, even a small amount, to alleviate the stress of frequent contact.

The combination of these factors creates a rich, probabilistic environment where the best collection strategy is not obvious and changes over time for each individual debtor.

### 2. The Multi-Armed Bandit (MAB) Framework
The MAB framework is used to decide which debtor to visit next. Each debtor is an "arm" of a multi-armed slot machine. Pulling an arm (visiting) yields a "reward" (money collected). The goal is to maximize the total reward over time.

#### **Upper Confidence Bound (UCB)**
UCB is an "optimism in the face of uncertainty" algorithm. It assigns a score to each debtor and chooses the one with the highest score.

* **The Formula**:
    $$\text{UCB Score} = \underbrace{\left(\frac{\text{Total Reward}}{\text{Visits}}\right)}_{\text{Average Performance}} + \underbrace{\sqrt{\frac{C \cdot \ln(\text{Total Visits})}{\text{Visits}}}}_{\text{Uncertainty Bonus}}$$

* **Explanation**:
    1.  **Average Performance (Exploitation)**: The first part of the formula is the historical average reward from that debtor. This makes the algorithm want to **exploit** debtors who have been reliable payers.
    2.  **Uncertainty Bonus (Exploration)**: The second part is a bonus for uncertainty. If a debtor has been visited very few times (`Visits` is small), this term becomes very large. This encourages the algorithm to **explore** debtors about whom it has little information, because they *might* be a hidden gem. As a debtor is visited more, this bonus shrinks, and the algorithm relies more on their proven performance.

#### **Thompson Sampling**


* **The Model**: Instead of just tracking an average, it models the potential of each debtor as a full probability distribution (a Beta distribution). This distribution is defined by two parameters: `alpha` (successes) and `beta` (failures).
    ```python
    class ThompsonSamplingCollector:
        # ...
        self.alpha = {debtor_id: 1.0 for debtor_id in debtor_ids}  # Successes + 1
        self.beta = {debtor_id: 1.0 for debtor_id in debtor_ids}   # Failures + 1
    ```
* **The Decision Process**:
    1.  **Sample**: For every potential debtor, it draws one random number from their personal Beta distribution.
    2.  **Select**: It chooses to visit the debtor who produced the highest random number in that round.
    3.  **Update**: After the visit, if it was successful (money collected), it increases the debtor's `alpha` parameter. If it failed, it increases `beta`. This update refines the shape of the distribution, making future samples more accurate.

This process balances exploration and exploitation. A new debtor has a flat, wide distribution, giving it a chance to produce a high random sample (exploration). A proven debtor has a tall, narrow distribution skewed towards high values, making it a consistently strong candidate (exploitation).

---

## Phase 4: The Experiment & Statistical Validation

The final phase rigorously tests the two MAB algorithms against each other to find a definitive winner.

* **Methodology**: The simulation is run **1,000 times** for both UCB and Thompson Sampling. Crucially, each run uses a new, randomly generated economic profile for the debtor population (varying average incomes and distributions). This ensures that the results are robust and not just an artifact of one specific setup.
* **Statistical Validation**: The notebook performs a **Null Hypothesis Test**.
    * **The Null Hypothesis ($H_0$)**: The initial assumption is that there is *no real performance difference* between UCB and Thompson Sampling. Any observed difference is just due to random chance.
    * **The Result**: The statistical tests yield a high **p-value** (e.g., `p=0.7752`). A p-value greater than 0.05 means we **cannot reject the null hypothesis**. In simple terms, despite one algorithm having a slightly higher average collection, the difference is not statistically significant. We cannot confidently declare a winner based on this experiment; they are both comparably effective.
* **Final Insights**: The visualizations, particularly the **Debt Recovery Timeline**, show that both algorithms learn and adapt effectively. They exhibit a classic S-curve of performance: a slow start during the initial exploration phase, a rapid increase in collections during the exploitation phase, and a final plateau as the easiest debts are collected.
* **Insights**: It is useful to generate insights from the current scenario and helps answer questions like - How much time to recover 50% debt in the current market trends (the probablistic assumptions which can be derived from real data)?

## Further work
This can be extended to model any new debtor class with specifc assumptions to get insights on profatibility and break even point.
Implementation of Reinforcement Learning to minimise travel time and number of visits to reduce logistics and labour costs.
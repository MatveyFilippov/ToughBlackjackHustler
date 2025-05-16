from environment import GameAction
from narrower_for_default_game import DefaultGameStateWithNarrowedProbability, QTableNarrower4DefaultGame
from learning_engine.q_learning import QValue
from history import Q_TABLES_HISTORY
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


q_table_history: dict[int, dict[DefaultGameStateWithNarrowedProbability, dict[GameAction, QValue]]] = {
    i: QTableNarrower4DefaultGame(q_table).weight_average_by_distance().to_dict()
    for i, q_table in Q_TABLES_HISTORY.items()
}


q_table_last_gen = q_table_history[max(q_table_history.keys())]


########################################################################################################################
# Distribution of Q-Values
plt.figure(figsize=(10, 5))
sns.histplot(
    [val for act_val in q_table_last_gen.values() for val in act_val.values() if val != QValue.NEUTRAL],
    kde=True, bins=30
)
plt.title("Distribution of Q-Values on the Last Generation")
plt.xlabel("Q-Value")
plt.ylabel("Frequency")
plt.show()
########################################################################################################################


q_val_hit_last_gen = []
q_val_stand_last_gen = []
for act_val in q_table_last_gen.values():
    for act, val in act_val.items():
        if act == GameAction.HIT:
            q_val_hit_last_gen.append(val)
        elif act == GameAction.STAND:
            q_val_stand_last_gen.append(val)
        else:
            raise ValueError("Unknown GameAction for DefaultGame")


########################################################################################################################
# Distribution of Q-Values by Action
plt.figure(figsize=(8, 6))
sns.boxplot(data=[q_val_hit_last_gen, q_val_stand_last_gen], palette="Set2")
plt.xticks([0, 1], [GameAction.HIT.name, GameAction.STAND.name])
plt.ylabel("Q-Value")
plt.title("Distribution of Q-Values by GameAction on the Last Generation")
plt.show()
########################################################################################################################


########################################################################################################################
# Box and Whisker Plot for Q-Values for HIT & Q-Values for STAND
plt.figure(figsize=(8, 8))
plt.scatter(q_val_hit_last_gen, q_val_stand_last_gen, alpha=0.5)
plt.xlabel(f"Q({GameAction.HIT.name})")
plt.ylabel(f"Q({GameAction.STAND.name})")
plt.title(f"Q({GameAction.HIT.name}) vs Q({GameAction.STAND.name}) on the Last Generation")
plt.plot([-1, 1], [-1, 1], linestyle="--", color="red")
plt.grid(True)
plt.show()
########################################################################################################################


player_sum = [s.player_cards_sum for s in q_table_last_gen]
dealer_card = [s.dealer_open_card for s in q_table_last_gen]


########################################################################################################################
# Visualize simple strategies (separate)
fig = plt.figure(figsize=(20, 8))

ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(player_sum, dealer_card, q_val_hit_last_gen, c=q_val_hit_last_gen, cmap='viridis')
ax1.set_xlabel("Player Sum")
ax1.set_ylabel("Dealer Card")
ax1.set_zlabel(f"Q({GameAction.HIT.name})")
ax1.set_title(f"Q-Values for {GameAction.HIT.name} on the Last Generation")

ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(player_sum, dealer_card, q_val_stand_last_gen, c=q_val_stand_last_gen, cmap='plasma')
ax2.set_xlabel("Player Sum")
ax2.set_ylabel("Dealer Card")
ax2.set_zlabel(f"Q({GameAction.STAND.name})")
ax2.set_title(f"Q-Values for {GameAction.STAND.name} on the Last Generation")

plt.tight_layout()
plt.show()
########################################################################################################################


########################################################################################################################
# Visualize simple strategies (together)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    player_sum, dealer_card, q_val_hit_last_gen,
    c='blue', label=GameAction.HIT.name, alpha=0.6, depthshade=False
)
ax.scatter(
    player_sum, dealer_card, q_val_stand_last_gen,
    c='red', label=GameAction.STAND.name, alpha=0.6, depthshade=False
)
ax.set_xlabel("Player Sum")
ax.set_ylabel("Dealer Card")
ax.set_zlabel("Q-Value")
ax.set_title(f"Comparison of Q-Values for {GameAction.HIT.name} and {GameAction.STAND.name} on the Last Generation")
ax.legend()
plt.show()
########################################################################################################################


########################################################################################################################
# Look at the average Q-Values across all historical generations
plt.figure(figsize=(10, 5))
plt.plot([
    sum(val for act_val in q_table.values() for val in act_val.values() if val != QValue.NEUTRAL) / (len(q_table) * 2)  # 2 GameAction on state
    for q_table in q_table_history.values()
])
plt.title("Average Q-Value for all GameStates in History")
plt.xlabel("Generation")
plt.ylabel("Average Q-Value")
plt.grid(True)
plt.show()
########################################################################################################################


########################################################################################################################
# Explanation how each generation influenced the agent's development
finding_new_states = []
strategy_changes = []
prev_optimal = None

for q_table in q_table_history.values():
    current_optimal = {state: max(actions.items(), key=lambda x: x[1])[0] for state, actions in q_table.items()}
    if prev_optimal:
        new_states = 0
        changes = 0
        for state in current_optimal:
            if state not in prev_optimal:
                new_states += 1
            elif current_optimal[state] != prev_optimal[state]:
                changes += 1
        finding_new_states.append(new_states)
        strategy_changes.append(changes)
    prev_optimal = current_optimal

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(finding_new_states)
plt.title("Percentage of Brand-New-State Findings between Generations")
plt.xlabel("Generation")
plt.ylabel("% of States Found")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(strategy_changes)
plt.title("Percentage of State-Action Changes between Generations")
plt.xlabel("Generation")
plt.ylabel("% of Changed Decisions")
plt.grid(True)

plt.show()
########################################################################################################################


########################################################################################################################
# Convergence graph for different actions
generations_q_values_on_hit = []
generations_q_values_on_stand = []
for gen in sorted(q_table_history.keys()):
    hit_vals = []
    stand_vals = []
    for act_val in q_table_history[gen].values():
        hit_vals.append(act_val[GameAction.HIT])
        stand_vals.append(act_val[GameAction.STAND])
    generations_q_values_on_hit.append(np.var(hit_vals))
    generations_q_values_on_stand.append(np.var(stand_vals))

plt.figure(figsize=(10, 5))
plt.plot(generations_q_values_on_hit, label=GameAction.HIT.name, color="blue")
plt.plot(generations_q_values_on_stand, label=GameAction.STAND.name, color="red")
plt.title("Variance of Q-Values Over Generations")
plt.xlabel("Generation")
plt.ylabel("Variance")
plt.legend()
plt.grid(True)
plt.show()
########################################################################################################################


########################################################################################################################
# Comparison of initial, middle and final Q-values on Violin Plot
q_values_by_gen = []
generations_used = []
for generation in sorted(q_table_history.keys())[::len(q_table_history)//5]:  # Lost some generations for plot bettor looking
    q_values_by_gen.append([
        val for act_val in q_table_history[generation].values() for val in act_val.values()
    ])
    generations_used.append(generation)

plt.figure(figsize=(20, 6))
sns.violinplot(data=q_values_by_gen, palette="muted")
plt.xticks(list(range(len(generations_used))), list(f"Gen{i}" for i in generations_used))
plt.title("Distribution of Q-Values: First - Middle - Last Generation")
plt.ylabel("Q-Value")
plt.grid(True)
plt.show()
########################################################################################################################

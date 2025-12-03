import pandas as pd
import math

# read data
cleaned_data = pd.read_csv("Adult_clean.csv")

# --- split training and testing set (80/20) ---
# shuffle data
shuffled_data = cleaned_data.sample(frac=1.0, random_state=42).reset_index(drop=True)

# split data by index
split_idx = int(0.8 * len(shuffled_data))
train = shuffled_data.iloc[:split_idx].copy()
test  = shuffled_data.iloc[split_idx:].copy()

# define income vs other feature
income   = "income"

features = []
for col in cleaned_data.columns:
    if col != "income":
        features.append(col)
        

# --- Naive Bayes model with income as single parent ---
# calculate prior over income
income_count = train[income].value_counts()
train_len = len(train)

# p_income: income_value -> probability
p_income = (income_count / train_len).to_dict() 

# states for each feature
feature_state = {}
for f in features:
    val = train[f].unique()
    feature_state[f] = sorted(val)
income_state  = sorted(train[income].unique())

# CPT: CPT[feature][income_value][feature_value] = probability
CPT = {}
for f in features:
    CPT[f] = {}
    val_f = feature_state[f]

    for i in income_state:
        # find rows where income = i
        income_group = train[train[income] == i]
        total_i = len(income_group)

        # freq of each value of feature f within a income group
        f_given_i = income_group[f].value_counts()

        # MLE estimate
        p_v_given_i = {}
        for v in val_f:
            count_vy = f_given_i.get(v, 0)
            if total_i > 0:
                p_v_given_i[v] = count_vy / total_i
            else:
                # if there is no obs for income value
                p_v_given_i[v] = 0.0

        CPT[f][i] = p_v_given_i


def posterior_income_nb(row, p_income, CPT, features):
    """
    Compute P(income | row) for the Naive Bayes model.
    Returns a dict like {'<=50K': p1, '>50K': p2}.
    """
    income_states = list(p_income.keys())
    log_scores = {}

    for y in income_states:
        logp = math.log(max(p_income[y], 1e-12))

        for f in features:
            v = row[f]
           
            p_f_given_y = CPT[f][y].get(v, 1e-9)
            logp += math.log(max(p_f_given_y, 1e-12))

        log_scores[y] = logp

    max_log = max(log_scores.values())
    exps = {y: math.exp(log_scores[y] - max_log) for y in income_states}
    Z = sum(exps.values())
    return {y: exps[y] / Z for y in income_states}

def postTest(test_df, condition_fn, p_income, CPT, features):
    """
    condition_fn: function that takes df and returns a boolean mask.
    Example: lambda df: df["education"] == "Bachelors"
    """
    subset = test_df[condition_fn(test_df)]
    if len(subset) == 0:
        return 0, None 

    values = []
    for _, row in subset.iterrows():
        post = posterior_income_nb(row, p_income, CPT, features)
        values.append(post['>50K'])

    return len(subset), sum(values) / len(values)


n_bach, p_bach = postTest(
    test,
    lambda df: df["education"] == "Bachelors",
    p_income, CPT, features
)

n_grad, p_grad = postTest(
    test,
    lambda df: df["education"].isin(["Masters", "Doctorate"]),
    p_income, CPT, features
)

n_ft, p_ft = postTest(
    test,
    lambda df: df["hours_bin"] == "Full-time",
    p_income, CPT, features
)

n_ext, p_ext = postTest(
    test,
    lambda df: df["hours_bin"] == "Extreme",
    p_income, CPT, features
)

n_exec, p_exec = postTest(
    test,
    lambda df: df["occupation"] == "Exec-managerial",
    p_income, CPT, features
)

print("education=Bachelors:", n_bach, p_bach)
print("education=Grad:", n_grad, p_grad)
print("hours_bin=Full-time:", n_ft, p_ft)
print("hours_bin=Extreme:", n_ext, p_ext)
print("occupation=Exec-managerial:", n_exec, p_exec)

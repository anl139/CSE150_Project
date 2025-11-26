import pandas as pd

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
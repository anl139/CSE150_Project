from ucimlrepo import fetch_ucirepo
import pandas as pd

adult = fetch_ucirepo(id=2)

X = adult.data.features                                     # 48842 x 14
y = adult.data.targets                                      # 48842 x 1 (DataFrame with 'income')

df = pd.concat([X, y], axis=1)                              # Combine into one DataFrame (48842 x 15) 

cols_to_drop = [
    'fnlwgt', 
    'education-num', 
    'capital-gain', 
    'capital-loss'
]

df = df.drop(columns=cols_to_drop)                          # Drops Unnecessary/Redundant Columns

df = df.replace(["?", "", " "], pd.NA)                      # Replace ?, "", " " with <NA>

for col in ['workclass', 'occupation', 'native-country']:   # Replace with 'Missing'
    df[col] = df[col].fillna('Missing')

df['income'] = df['income'].str.strip()                     # remove spaces and '.' from income
df['income'] = df['income'].str.replace('.', '', regex=False)

### Group Native-Countries to Reigions ###

latin_america = [
    'Mexico', 'Puerto-Rico', 'Honduras', 'Jamaica', 'Trinadad&Tobago',
    'Nicaragua', 'Guatemala', 'El-Salvador', 'Columbia', 'Ecuador', 'Peru',
    'Dominican-Republic', 'Haiti'
]

asia = [
    'India', 'China', 'Japan', 'Philippines', 'Vietnam', 'Thailand',
    'Hong', 'Taiwan', 'Cambodia', 'Laos'
]

europe = [
    'England', 'Germany', 'Italy', 'Poland', 'Portugal', 'Ireland',
    'France', 'Scotland', 'Greece', 'Hungary', 'Holand-Netherlands',
    'Yugoslavia'
]

def CountryToRegion(x):                                 # Groups Native-Countries to Regions
    if x == 'Missing':
        return 'Missing'
    if x == 'United-States':
        return 'US'
    if x in latin_america:
        return 'Latin-America'
    if x in asia:
        return 'Asia'
    if x in europe:
        return 'Europe'
    return 'Other'  

df['native-country'] = df['native-country'].apply(CountryToRegion)

### Group Education to Education Levels ###

def EducationLevel(e):
    HS_Less = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th']
    some_college = ['Some-college', 'Assoc-acdm', 'Assoc-voc']
    grad = ['Masters', 'Doctorate', 'Prof-school']

    if e in HS_Less:                                # Groups Education to Education Levels
        return 'HS-or-less'
    if e == 'HS-grad':
        return 'HS-grad'
    if e in some_college:
        return 'Some-college/Assoc'
    if e == 'Bachelors':
        return 'Bachelors'
    if e in grad:
        return 'Grad'
    return e  # fallback

df['education'] = df['education'].apply(EducationLevel)

### Groups Workclass to Workclass Levels ###

def map_workclass(w):                               # Groups Workclass to Workclass Levels
    gov = ['Federal-gov', 'Local-gov', 'State-gov']
    private_self = ['Private', 'Self-emp-inc', 'Self-emp-not-inc']
    not_working = ['Without-pay', 'Never-worked']

    if w == 'Missing':
        return 'Missing'
    if w in gov:
        return 'Gov'
    if w in private_self:
        return 'Private/Self-emp'
    if w in not_working:
        return 'Not-working'
    return w  # fallback

df['workclass'] = df['workclass'].apply(map_workclass)

### Make continous Variables Discrete ###

df['age_bin'] = pd.cut(                                     # Changed Age to Discrete
    df['age'],
    bins=[0, 25, 45, 65, 100],
    labels=['Young', 'Adult', 'Middle', 'Senior'],
    include_lowest=True
)

df['hours_bin'] = pd.cut(                                   # Changed Hours Per Week to Discrete
    df['hours-per-week'],
    bins=[0, 30, 40, 60, 100],
    labels=['Part-time', 'Full-time', 'Long', 'Extreme'],
    include_lowest=True
)

df = df.drop(columns=['age', 'hours-per-week'])             # Drop Age and Hours-per-week columns
Adult_clean = df

output_path = "Adult_clean.csv"                             # Output Clean Data to "Adult_clean.csv"
Adult_clean.to_csv(output_path, index=False)

Adult_encoded = df.copy()                                   # Create Copy for Encoded

for col in Adult_encoded.columns:                           # Created Data Points to Integers for BN
    categories = Adult_encoded[col].astype('category').cat.categories
        
output_path1 = "Adult_encoded.csv"
Adult_encoded.to_csv(output_path1, index=False)


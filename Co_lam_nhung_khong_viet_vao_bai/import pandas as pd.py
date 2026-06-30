import pandas as pd

df = pd.read_csv('outputs/mendeley_train_clean.csv')
trait_cols = ['agreeableness','openness',
              'conscientiousness','extraversion','neuroticism']
print("=== MENDELEY STATS (sau chuẩn hóa 0-1) ===")
print(df[trait_cols].describe().round(3))
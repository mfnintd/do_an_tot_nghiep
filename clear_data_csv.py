import pandas as pd
data = {
  "distance": [],
  "camera_height": [],
  "person_height": [],
  "type": [],
  "h1": [],
  "h2": [],
  "h3": [],
  "h4": [],
  "h5": []
}
df = pd.DataFrame(data)

df.to_csv('./data/data.csv', index=False)

print(df)
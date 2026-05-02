"""
Random sampler for NOW corpus modal verb data.
Takes 100 random rows from each of the 4 Excel files
and combines them into one annotation file.
"""

import pandas as pd

# List of files to sample
files = ["mustNOW.xlsx", "shouldNOW.xlsx", "deveNOW.xlsx", "precisaNOW.xlsx"]

# Empty list to hold all samples
all_samples = []

# Loop through each file
for file in files:
    df = pd.read_excel(file, header=None)        # read the file
    sample = df.sample(n=250, random_state=42)   # pick 100 random rows
    all_samples.append(sample)                   # add to the list

# Combine all samples into one file
combined = pd.concat(all_samples)
combined["Annotation"] = ""                      # add empty column for annotation
combined.to_excel("samples_to_annotate.xlsx", index=False)

print("Done! 400 rows saved to samples_to_annotate.xlsx")

import sys
import pandas as pd

template_file = sys.argv[1]
structure_file = sys.argv[2]

# load templates
df_templates = pd.read_csv(template_file)

# build dictionary: EventId -> EventTemplate
x_dict = dict(zip(df_templates["EventId"], df_templates["EventTemplate"]))

df_struct = pd.read_csv(structure_file)

# update EventTemplate based on EventId
df_struct["EventTemplate"] = df_struct["EventId"].map(x_dict)

df_struct.to_csv(structure_file, index=False)

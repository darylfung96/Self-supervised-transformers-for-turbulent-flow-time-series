import numpy as np
import pandas as pd
import re

probe_num = '9'
with open(f'probe.000{probe_num}.his', 'r') as f:
    content = f.read()

content = content.split('\n')[:-1]
content = [re.sub(r'\s+', ',', item.strip()) for item in content]
content = [item.split(',') for item in content]
content = [[item[0], item[-1]] for item in content]
content = np.array(content).astype(float)

df = pd.DataFrame(content)
df.to_csv(f'probe(0{probe_num},1).csv', index=False)

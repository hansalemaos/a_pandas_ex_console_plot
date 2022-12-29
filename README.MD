<h2>Retro Pandas - Plotting back in 1985</h2>


<img src="https://decider.com/wp-content/uploads/2014/07/back-to-the-future.jpg?quality=90&strip=all&w=646&h=335&crop=1"/>



```python

$pip install a-pandas-ex-console-plot

from a_pandas_ex_console_plot import pd_add_console_plot
import pandas as pd
import numpy as np
pd_add_console_plot()
#https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
df2 = pd.DataFrame(np.random.rand(10, 4), columns=["a", "b", "c", "d"])
df2.ds_console_plot(sizepic=70, kind="bar", subplots=False)

```
### Isn't that beautiful? hahahaha

<img src="https://github.com/hansalemaos/screenshots/raw/main/pandasconsoleplott.png"/>


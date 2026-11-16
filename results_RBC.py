import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parent

filename = BASE / "macro_simulation.csv"
df = pd.read_csv(filename)[0:1000]
t = np.array(df["t"])[0:1000]
capital = np.array(df["Capital"])[0:1000]
realGDP = np.array(df["Real GDP"])[0:1000]
consumption = np.array(df["Consumption"])[0:1000]
investment = np.array(df["Investment"])[0:1000]
productivity = np.array(df["Productivity "])[0:1000]

t = t / 12

fig, axes = plt.subplots(nrows=2, ncols=2)

axes[0,0].plot(t,capital, color="red")
axes[0,1].plot(t,realGDP, color="navy")
axes[1,0].plot(t,consumption, color="purple")
axes[1,1].plot(t,investment, color="green")

axes[0,0].set_xlabel("Time (years)")
axes[0,1].set_xlabel("Time (years)")
axes[1,0].set_xlabel("Time (years)")
axes[1,1].set_xlabel("Time (years)")

axes[0,0].set_ylabel("Capital (USD Trillion)")
axes[0,1].set_ylabel("Real GDP (USD Trillion)")
axes[1,0].set_ylabel("Consumption (USD Trillion)")
axes[1,1].set_ylabel("Investment (USD Trillion)")

axes[0,0].set_title("Capital vs Time")
axes[0,1].set_title("Real GDP vs Time")
axes[1,0].set_title("Consumption vs Time")
axes[1,1].set_title("Investment vs Time")
plt.show()
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np


x = np.linspace(-3, 3, 50)
y = np.exp(x) + 0.1*np.random.randn(50)
plt.plot(x, y, 'ro')

spl = UnivariateSpline(x, y)
xs = np.linspace(-3, 3, 100)
plt.plot(xs, spl(xs), 'g', lw=3)

plt.show()





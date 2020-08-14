## Get started

==========================================================

### Requirements

The code and examples of this book should work for Python 3.5+. 

Install the following Python packages

- NumPy (`pip install numpy`)
- SciPy (`pip install scipy`)
- MatplotLib (`pip install matplotlib`)
- Scikit-image (`pip install scikit-image`)

You should also have Jupyter notebook/lab environment installed and built for interactive practice.

### Testing after install

Open a Jupyter notebook and execute the following code,

```
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, io, filters

image = data.coins()  # or any NumPy array!
edges = filters.sobel(image)
io.imshow(edges)
```

You should see the following output. If you see this, you are all set to go!

```{figure} Images/sobel_coins.png
---
name: Sobel-coins
---
Sobel filter on coins' image
```





import numpy as np
import matplotlib.pyplot as plt
frame = np.zeros((576,720,3), dtype=np.uint8)
plt.imshow(frame)
plt.show()
with open("frame.rgb","wb") as f:
    f.write(frame.tobytes())
import cv2
import numpy as np
writer = cv2.VideoWriter("output.avi",
cv2.VideoWriter_fourcc(*"MJPG"), 30,(220,220))
image = np.random.randint(0, 255, (220,220,3)).astype('uint8')
for frame in range(1000):
    writer.write(image)

writer.release()
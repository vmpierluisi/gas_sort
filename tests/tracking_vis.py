from matplotlib import pyplot as plt
import cv2

img_13_1 = cv2.imread("../detections/MOT16-13-500.png", cv2.IMREAD_COLOR_RGB)
img_13_2 = cv2.imread("../detections/MOT16-13-520.png", cv2.IMREAD_COLOR_RGB)
img_13_3 = cv2.imread("../detections/MOT16-13-540.png", cv2.IMREAD_COLOR_RGB)

img_04_1 = cv2.imread("../detections/MOT16-04-100.png", cv2.IMREAD_COLOR_RGB)
img_04_2 = cv2.imread("../detections/MOT16-04-120.png", cv2.IMREAD_COLOR_RGB)
img_04_3 = cv2.imread("../detections/MOT16-04-140.png", cv2.IMREAD_COLOR_RGB)


fig = plt.figure(figsize=(24, 12))
plt.subplot(231); plt.imshow(img_04_1); plt.title("MOT17-04-Frame 100", fontsize=20)
plt.subplot(232); plt.imshow(img_04_2); plt.title("MOT17-04-Frame 120", fontsize=20)
plt.subplot(233); plt.imshow(img_04_3); plt.title("MOT17-04-Frame 140", fontsize=20)
plt.subplot(234); plt.imshow(img_13_1); plt.title("MOT17-13-Frame 500", fontsize=20)
plt.subplot(235); plt.imshow(img_13_2); plt.title("MOT17-13-Frame 520", fontsize=20)
plt.subplot(236); plt.imshow(img_13_3); plt.title("MOT17-13-Frame 540", fontsize=20)
fig.supxlabel("x-axis Coordinates", fontsize=20, y=0.04)
fig.supylabel("y-axis Coordinates", fontsize=20, x=0.01)
plt.tight_layout()
plt.savefig("../detections/MOT-example.png", dpi=300)
plt.show()



fig = plt.figure(figsize=(5, 8))
plt.subplot(311); plt.imshow(img_04_1); plt.title("Frame 100", fontsize=10)
plt.subplot(312); plt.imshow(img_04_2); plt.title("Frame 120", fontsize=10)
plt.subplot(313); plt.imshow(img_04_3); plt.title("Frame 140", fontsize=10)
fig.supxlabel("x-axis Coordinates", fontsize=10, y=0.02)
fig.supylabel("y-axis Coordinates", fontsize=10, x=0.05)
plt.tight_layout()
plt.savefig("../detections/MOT-example2.png", dpi=300)
plt.show()


fig = plt.figure(figsize=(22, 12))
plt.subplot(221); plt.imshow(img_04_1); plt.title("MOT17-04-Frame 100", fontsize=20)
plt.subplot(222); plt.imshow(img_04_2); plt.title("MOT17-04-Frame 120", fontsize=20)
#plt.subplot(223); plt.imshow(img_04_3); plt.title("MOT17-04-Frame 140", fontsize=20)
#plt.subplot(224); plt.imshow(img_13_1); plt.title("MOT17-13-Frame 500", fontsize=20)
plt.subplot(223); plt.imshow(img_13_2); plt.title("MOT17-13-Frame 520", fontsize=20)
plt.subplot(224); plt.imshow(img_13_3); plt.title("MOT17-13-Frame 540", fontsize=20)
fig.supxlabel("x-axis Coordinates", fontsize=20, y=0.02)
fig.supylabel("y-axis Coordinates", fontsize=20, x=0.02)
plt.tight_layout()
plt.savefig("../detections/MOT-example3.png", dpi=300)
plt.show()
import random
UPPER_BOUND = 100
with open("demo_img.txt", "w") as f:
    for i in range(0, UPPER_BOUND):
        f.write(str(int(random.random() * 7)) + "\n")

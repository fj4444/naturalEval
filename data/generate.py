with open("picture_name.txt", "w") as f:
    for i in range(0, 100):
        f.write("img" + str(i) + ".png" + "\n")
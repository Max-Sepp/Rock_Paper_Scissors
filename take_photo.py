import cv2
import time

camera_port = 0
camera = cv2.VideoCapture(camera_port)

mode = str(input("Please input the category of photo you are taking"))

while True:
    user_input = str(
        input(
            "input nothing if you want to take 50 photos over the next 10 seconds. Type 'Change Mode' to change category of photo you are taking or type anything else to escape and leave the program"
        )
    )
    if user_input == "":
        time.sleep(3)
        images_taken = 0
        while images_taken < 50:
            result, image = camera.read()

            if result:
                cv2.imwrite(f"{mode}_{time.time_ns()}.png", image)
            else:
                print("No image detected")
            time.sleep(0.2)
            images_taken += 1
    elif user_input == "Change Mode":
        mode = str(input("Please input the category of photo you are taking"))
    else:
        break

cv2.destroyAllWindows()

import threading

import cv2
import dlib
import mouse as ms
import numpy as np
import pyautogui as pm


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask


def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
    global counter, ix, iy, movex, movey
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        if counter != 10:
            ix = cx
            iy = cy
            counter += 1
        # + for left - for right
        # + for up - for down
        main_task(cx, cy)
        ms.move(movex, movey)
        #print(cx, ix, cy, iy)
    except:
        pass


def update_x(x):
    global ix, movex, xscr
    if ix > x and movex < xscr-20:
        movex += 5
    else:
        if movex > 20:
            movex -= 5


def update_y(y):
    global iy, movey
    if iy < y and movey < yscr-20:
        movey += 5
    else:
        movey -= 5


def main_task(x, y):
    # creating threads
    t1 = threading.Thread(target=update_x, args=(x, ))
    t2 = threading.Thread(target=update_y, args=(y, ))

    # start threads
    t1.start()
    t2.start()

    # wait until threads finish their job
    t1.join()
    t2.join()


counter, ix, iy, movex, movey = 0, 0, 0, 0, 0
i = 0
xscr, yscr = pm.size()
ms.move(xscr / 2, yscr / 2)
#pid = os.getpid()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

kernel = np.ones((9, 9), np.uint8)


def nothing(x):
    pass


while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    for rect in rects:
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask = eye_on_mask(mask, left)
        mask = eye_on_mask(mask, right)
        mask = cv2.dilate(mask, kernel, 5)
        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = (shape[42][0] + shape[39][0]) // 2
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = 250
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = cv2.erode(thresh, None, iterations=2)  # 1
        thresh = cv2.dilate(thresh, None, iterations=4)  # 2
        thresh = cv2.medianBlur(thresh, 3)  # 3
        thresh = cv2.bitwise_not(thresh)
        contouring(thresh[:, 0:mid], mid, img)
        contouring(thresh[:, mid:], mid, img, True)
    flipHorizontal = cv2.flip(img, 1)
    cv2.imshow('eyes', flipHorizontal)
    if cv2.waitKey(1) & 0xFF == ord('q'): # & 0xFF
        break
    i = 0
    while i in range(2):
        if cv2.waitKey(1) & 0xFF == ord('h'):
            while True:
                print("Hold!!!")
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    pm.click(button='right')
                    print("RIGHT BUTTON PRESSED")
                    break
                if cv2.waitKey(1) & 0xFF == ord('l'):
                    pm.click(button='left')
                    print("LEFT BUTTON PRESSED")
                    break
        i += 1
cap.release()
cv2.destroyAllWindows()

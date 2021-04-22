import cv2
import numpy as np

backSub = cv2.createBackgroundSubtractorKNN()


# Callback function for trackbar
def on_change(self):
    pass


def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0, 0), sigma) + 127


def main():
    cap = cv2.VideoCapture("mt_camera_raw.avi")
    image = cv2.imread("background.jpg")

    create_trackbars()

    while cap.isOpened():
        _, frame = cap.read()

        # Catches function from crashing if video ends
        # Currently used to loop the video indefinitely (press wait key to exit program)
        if frame is None:
            cap = cv2.VideoCapture("mt_camera_raw.avi")
            _, frame = cap.read()

        # convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # BG subtraction
        frame = bg_subtraction(frame)

        # blur
        frame = cv2.GaussianBlur(frame, (11, 11), 0)

        # apply highpass
        frame = highpass(frame, 3)

        # apply threshold
        frame, threshold = brightness_threshold(frame)

        find_contours(frame, threshold)

        # Display results
        cv2.imshow('frame', frame)

        # Determine replay speed and terminate if "Q"
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def create_trackbars():
    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("threshold", "Trackbars", 140, 255, on_change)
    cv2.createTrackbar("min_area", "Trackbars", 10, 500, on_change)
    cv2.createTrackbar("max_area", "Trackbars", 100, 500, on_change)


def brightness_threshold(frame):
    threshold_value = cv2.getTrackbarPos("threshold", "Trackbars")
    _, threshold = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)
    #  apply the threshold mask to the frame (not actually sure if needed)
    frame = cv2.bitwise_and(frame, frame, mask=threshold)
    return frame, threshold


def find_contours(frame, threshold):
    min_area = cv2.getTrackbarPos("min_area", "Trackbars")
    max_area = cv2.getTrackbarPos("max_area", "Trackbars")

    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:  # if there is any contour
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:  # check if the contour area is within the set boundaries
                x, y, w, h = cv2.boundingRect(contour)
                #  cv2.circle(frame, center, radius, color(BRG), thickness)
                cv2.circle(frame, (x, y), int((w + h) / 2), (255, 0, 0), 2)


def bg_subtraction(frame):
    foreground_mask = backSub.apply(frame)
    frame = cv2.bitwise_and(frame, frame, mask=foreground_mask)
    return frame


if __name__ == '__main__':
    main()

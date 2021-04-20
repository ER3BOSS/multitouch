import cv2
import numpy as np

backSub = cv2.createBackgroundSubtractorKNN()


# Callback function for trackbar
def on_change(self):
    pass


def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0, 0), sigma) + 127


def run():
    cap = cv2.VideoCapture("mt_camera_raw.avi")
    image = cv2.imread("background.jpg")

    cv2.namedWindow("Trackbars")
    cv2.createTrackbar("threshold", "Trackbars", 180, 255, on_change)
    # value = cv2.getTrackbarPos('threshold', 'Trackbars')

    while cap.isOpened():
        _, frame = cap.read()
        # Catches function from crashing if video ends
        if frame is None:
            break

        # convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # BG subtraction
        frame = bg_subtraction(frame)

        # apply highpass
        frame = highpass(frame, 3)

        # apply threshold
        threshold_value = cv2.getTrackbarPos("threshold", "Trackbars")
        _, frame = cv2.threshold(frame, threshold_value, 255, cv2.THRESH_BINARY)

        # Display results
        cv2.imshow('frame', frame)

        # Determine replay speed and terminate if "Q"
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def bg_subtraction(frame):
    foreground_mask = backSub.apply(frame)
    frame = cv2.bitwise_and(frame, frame, mask=foreground_mask)
    return frame


def main():
    run()


if __name__ == '__main__':
    main()

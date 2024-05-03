import cv2
from ultralytics import YOLO
import numpy as np 

# Load the YOLOv8 model
model = YOLO('best-seg-test.pt')

# Open the video file
video_path = r'\Users\ninth\Documents\code\SEGMENTATIONMODEL\aimodeltestvid.mp4'
cap = cv2.VideoCapture(video_path)

count = 1

def convertToBinaryMatrix(image):
    """
    Sets pixels with high R channel values to R = 255, otherwise sets pixel color to black.
    Parameters
    ----------
    image : numpy.ndarray

    Returns
    -------
    binary_imaage : numpy.ndarray

    """
    
    # Make a copy of the image
    binary_image = np.copy(image) 

    # Extract the red channel
    red_channel = binary_image[:, :, 2]

    # Set pixels with R > 120 to (0, 0, 255) and others to (0, 0, 0)
    binary_image[red_channel > 120] = [0, 0, 255]
    binary_image[red_channel <= 120] = [0, 0, 0]

    return binary_image


# Loop through the video frames
while cap.isOpened():
    
    
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot(boxes=False, labels=False)

        # Display the annqotated frame
        #cv2.imshow("YOLOv8 Inference", annotated_frame)
    
        cv2.imwrite("frame%d.png" % count, convertToBinaryMatrix(annotated_frame))

        # Save the frame to working folder
        #cv2.imwrite("frame%d.png" % count, annotated_frame)
        
        count += 1 

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
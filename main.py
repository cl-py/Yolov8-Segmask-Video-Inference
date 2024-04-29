import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('best-seg-test.pt')

# Open the video file
video_path = r'\Users\ninth\Documents\code\SEGMENTATIONMODEL\aimodeltestvid.mp4'
cap = cv2.VideoCapture(video_path)


def convertToBinaryMatrix(image):
    """
    Loops through all pixels in the image, locating pixels with high R values and
    assigning those pixels to R = 255. Otherwise, assigns to BLACK.
    Parameters
    ----------
    image : Frame from YOLOv8 results.plot.
        Contains the processed frame retrieved after running YOLOv8 inference.

    Returns
    -------
    rgb_values : TYPE
        DESCRIPTION.

    """

    width, height = image.size
    
    
    # Iterates through entire image pixel by pixel.
    for y in range(height):
        
        for x in range(width):
            
            pixel_value = annotated_frame[200, 100]
            
            # Appends binary pixel values to a 2D array. 
            
            #if pixel_value[0] > 120:
                #rgb_values.append(0)
                
            #else:
                #rgb_values.append(1)
                
    return image


# Loop through the video frames
while cap.isOpened():
    
    count = 1
    
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot(boxes=False, labels=False)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        
        pixel_value = annotated_frame[200, 100]

        print("RGB values at (100, 200):", pixel_value)
    
        #annotated_frame = convertToBinaryMatrix(annotated_frame)

        #Save the frame to working folder
        cv2.imwrite("frame%d.png" % count, annotated_frame)
        
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
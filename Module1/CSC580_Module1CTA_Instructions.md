# Option #1: Face Detection
For this assignment, you will write Python code to detect faces in an image. Use the following Python code as a starter for your program.

Supply your own image file that contains one or more faces to identify. 
An example output for your program should be something like the following picture with red boxes (instead of green) drawn around each individual’s face:

Code Steps
1. import PIL.ImageDraw
2. import face_recognition
3. Load the jpg file into a numpy array
4. Find all the faces in the image # Use the following Python pseudocode as guidance for your solution.
   - numberOfFaces = len(faceLocations)print("Found {} face(s) in this picture.".format(numberOfFaces))
5. Load the image into a Python Image Library object so that you can draw on top of it and display it
   - pilImage = PIL.Image.fromarray(image)
6. for faceLocation in faceLocations:
   - Print the location of each face in this image. Each face is a list of co-ordinates in (top, right, bottom, left) order.
   - print("A face is located at pixel location Top: {}, Left {},Bottom: {}, Right: {}".format(top, left, bottom, right))
   - Draw a box around the face     drawHandle = PIL.ImageDraw.Draw(pilImage)     drawHandle.rectangle([left, top, right, bottom], outline="red")
   - Display the image on screenpilImage.show()

Develop the remaining code in the section specific to face detection. 
Because most human faces have roughly the same structure, the pre-trained face detection model will work well for almost any image. 
There's no need to train a new one from scratch. Use PIL, which is the Python Image Library.


# Module 1 Critical Thinking Assignment
# Option 1: Face Detection 

# Alexander Reichart-Anderson
# MS in AI and ML, Colorado State University Global
# CSC580-1: Applying Machine Learning & Neural Networks - Capstone
# Dr. Joseph Issa
# May 18, 2025

# ------------------------------
# To Install OpenCV: pip install face_recognition
# ------------------------------

# Requirement 0A: Import Required Packages
import face_recognition
import PIL.Image
import PIL.ImageDraw

# Requirement 1: Load the image file into a numpy array
image_path = "Module1/faces_in_the_crowd.jpg" 
image = face_recognition.load_image_file(image_path)

# Requirement 2: Find all the faces in the image
face_locations = face_recognition.face_locations(image)

# Requirement 3: Print the number of faces found
number_of_faces = len(face_locations)
print(f"Found {number_of_faces} face(s) in this picture.")

# Requirement 4: Load the image into a PIL object so we can draw on it
pil_image = PIL.Image.fromarray(image)

# Requirement 5: Create function that draws boxes around identified faces
draw = PIL.ImageDraw.Draw(pil_image)

# Requirement 6: Loop through each face found in the image
for face_location in face_locations:
    # Requirement 6A: Identifiy faces and their pixel locations
    top, right, bottom, left = face_location
    print(f"A face is located at pixel location Top: {top}, Left: {left}, Bottom: {bottom}, Right: {right}")
    # Requirement 6B: Draw a red box around the face
    draw.rectangle([left, top, right, bottom], outline="red", width=3)

# Requirement 7: Create new image with boxes drawn around faces
pil_image.show()

# Module 4 Portfolio Milestone
# Option 1: Implementing Facial Recognition

# Alexander Reichart-Anderson
# MS in AI and ML, Colorado State University Global
# CSC580-1: Applying Machine Learning & Neural Networks - Capstone
# Dr. Joseph Issa
# June 15, 2025

# Step 0: Load Required Packages
import face_recognition

def is_individual_in_group():
    # Step 1A: Load Single Face Image (hardcoded path)
    individual_image = face_recognition.load_image_file('Module4/PM/images/individual.jpg')
    individual_encodings = face_recognition.face_encodings(individual_image)
    
    # Step 1B: Error Handling
    if len(individual_encodings) != 1:
        raise ValueError("Individual image must contain exactly one face")
    
    # Step 2A: Load Group Image (hardcoded path)
    group_image = face_recognition.load_image_file('Module4/PM/images/group.jpg')
    group_encodings = face_recognition.face_encodings(group_image)
    
    # Step 2B: Error Handling
    if not group_encodings:
        print("No faces found in the group image.")
        return False
    
    # Step 3: Compare individual encoding with all group encodings
    matches = face_recognition.compare_faces(group_encodings, individual_encodings[0])
    return any(matches)

if __name__ == "__main__":
    try:
        result = is_individual_in_group()
        print(f"Individual present in group: {result}")
    except Exception as e:
        print(f"Error: {str(e)}")

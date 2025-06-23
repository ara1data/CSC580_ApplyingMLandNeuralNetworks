# Module 6 Critical Thinking Assignment
# Option 1: Implementation of CIFAR10 with CNNs Using TensorFlow

# Alexander Reichart-Anderson
# MS in AI and ML, Colorado State University Global
# CSC580-1: Applying Machine Learning & Neural Networks - Capstone
# Dr. Joseph Issa
# June 22, 2025

# Step 0: Import Required Packages
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.utils import to_categorical


class CIFAR10CNN:
    def __init__(self):
        self.model = None
        self.history = None
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                           'dog', 'frog', 'horse', 'ship', 'truck']
        
    def load_and_preprocess_data(self):
        """
        Load the CIFAR-10 dataset and normalize the data
        Following flowchart: Load Dataset → Read Dataset → Normalize Dataset
        """
        print("Loading CIFAR-10 dataset...")
        
        # Load the dataset
        (self.X_train, self.y_train), (self.X_test, self.y_test) = datasets.cifar10.load_data()
        
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Training labels shape: {self.y_train.shape}")
        print(f"Test data shape: {self.X_test.shape}")
        print(f"Test labels shape: {self.y_test.shape}")
        
        # Normalize pixel values to be between 0 and 1
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0
        
        # Convert labels to categorical (one-hot encoding)
        self.y_train_categorical = to_categorical(self.y_train, 10)
        self.y_test_categorical = to_categorical(self.y_test, 10)
        
        print("Data preprocessing completed.")
        
    def visualize_sample_data(self):
        """
        Visualize sample images from the dataset
        """
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.X_train[i])
            plt.xlabel(self.class_names[self.y_train[i][0]])
        plt.suptitle('Sample CIFAR-10 Images')
        plt.tight_layout()
        plt.show()
        
    def define_cnn_architecture(self):
        """
        Define the Convolutional Neural Network architecture
        Following flowchart: Define CNN
        """
        print("Defining CNN architecture...")
        
        self.model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')  # 10 classes in CIFAR-10
        ])
        
        print("CNN architecture defined.")
        self.model.summary()
        
    def compile_model(self):
        """
        Define loss function and optimizer
        Following flowchart: Define Loss Function
        """
        print("Compiling model...")
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model compiled successfully.")
        
    def train_network(self, epochs=25, batch_size=32):
        """
        Train the network
        Following flowchart: Train Network
        """
        print(f"Training network for {epochs} epochs...")
        
        # Define callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.0001
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            self.X_train, self.y_train_categorical,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(self.X_test, self.y_test_categorical),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed.")
        
    def evaluate_model(self):
        """
        Test the network based on trained data
        Following flowchart: Test Network Based on Trained Data
        """
        print("Evaluating model on test data...")
        
        test_loss, test_accuracy = self.model.evaluate(
            self.X_test, self.y_test_categorical, verbose=0
        )
        
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_accuracy, test_loss
        
    def make_predictions(self, num_samples=10):
        """
        Make predictions on test data
        Following flowchart: Make Prediction on Test Data
        """
        print("Making predictions on sample test data...")
        
        # Select random samples from test set
        indices = np.random.choice(len(self.X_test), num_samples, replace=False)
        sample_images = self.X_test[indices]
        sample_labels = self.y_test[indices]
        
        # Make predictions
        predictions = self.model.predict(sample_images)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Visualize predictions
        plt.figure(figsize=(15, 6))
        for i in range(num_samples):
            plt.subplot(2, 5, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(sample_images[i])
            
            actual_label = self.class_names[sample_labels[i][0]]
            predicted_label = self.class_names[predicted_classes[i]]
            confidence = np.max(predictions[i]) * 100
            
            color = 'green' if actual_label == predicted_label else 'red'
            plt.xlabel(f'Actual: {actual_label}\nPredicted: {predicted_label}\nConfidence: {confidence:.1f}%', 
                      color=color)
        
        plt.suptitle('Sample Predictions')
        plt.tight_layout()
        plt.show()
        
    def plot_training_history(self):
        """
        Plot training and validation accuracy/loss
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
    def save_model(self, filepath='cifar10_cnn_model.h5'):
        """
        Save the trained model
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Train the model first.")

def main():
    """
    Main function implementing the complete workflow from the flowchart
    """
    print("=== CIFAR-10 CNN Classification Implementation ===")
    print("Following the provided flowchart workflow\n")
    
    # Initialize the CNN classifier
    cifar_cnn = CIFAR10CNN()
    
    # Step 1: Load and preprocess data
    cifar_cnn.load_and_preprocess_data()
    
    # Visualize sample data
    cifar_cnn.visualize_sample_data()
    
    # Step 2: Define CNN architecture
    cifar_cnn.define_cnn_architecture()
    
    # Step 3: Define loss function (compile model)
    cifar_cnn.compile_model()
    
    # Step 4: Train the network
    cifar_cnn.train_network(epochs=30, batch_size=32)
    
    # Step 5: Test the network
    test_accuracy, test_loss = cifar_cnn.evaluate_model()
    
    # Step 6: Make predictions
    cifar_cnn.make_predictions()
    
    # Plot training history
    cifar_cnn.plot_training_history()
    
    # Save the model
    cifar_cnn.save_model()
    
    # Step 7: Model analysis for report
    print("\n=== Model Analysis Summary ===")
    print(f"Final Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Final Test Loss: {test_loss:.4f}")
    
    if test_accuracy > 0.80:
        print("Model Performance: Excellent (>80% accuracy)")
    elif test_accuracy > 0.70:
        print("Model Performance: Good (70-80% accuracy)")
    elif test_accuracy > 0.60:
        print("Model Performance: Fair (60-70% accuracy)")
    else:
        print("Model Performance: Needs improvement (<60% accuracy)")

if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Run the main workflow
    main()

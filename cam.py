import tensorflow.compat.v1 as tf
import numpy as np
import os
import cv2
import time

# Disable eager execution for TensorFlow 1 compatibility
tf.compat.v1.disable_eager_execution()

# Initialize the categories for classification
CATEGORIES = ["non_violence", "violence"]
print(CATEGORIES[0])
print(CATEGORIES[1])

# Start video capture
video = cv2.VideoCapture(0)
time.sleep(2)  # Allow time for the camera to warm up

# Directory paths
train_path = './data/train'
test_path = './data/test'

# Ensure required directories exist
if not os.path.exists(train_path):
    raise Exception(f"Training directory {train_path} does not exist.")
if not os.path.exists(test_path):
    os.makedirs(test_path)

# Load the pre-trained model
sess = tf.Session()
saver = tf.train.import_meta_graph('model/trained_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model/'))
graph = tf.get_default_graph()

# Get the required tensors from the graph
y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")

# Constants
image_size = 128
num_channels = 3
y_test_images = np.zeros((1, len(CATEGORIES)))

# Timer and counts
start_time = time.time()
interval = 30  # Interval in seconds
results_count = {"non_violence": 0, "violence": 0}

while True:
    # Capture a frame from the webcam
    grabbed, frame = video.read()
    if not grabbed:
        break

    # Display the live webcam feed
    cv2.imshow("Input", frame)

    # Save a frame for testing every interval
    current_time = time.time()
    if current_time - start_time >= interval:
        # Reset the timer
        start_time = current_time

        # Save the frame to the test directory
        test_image_path = os.path.join(test_path, "test.jpg")
        cv2.imwrite(test_image_path, frame)

        # Preprocess the saved image
        if os.path.exists(test_image_path):
            image = cv2.imread(test_image_path)
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
            image = np.array(image, dtype=np.float32) / 255.0
            x_batch = image.reshape(1, image_size, image_size, num_channels)

            # Run the model for prediction
            feed_dict_testing = {x: x_batch, y_true: y_test_images}
            result = sess.run(y_pred, feed_dict=feed_dict_testing)

            # Interpret results
            confidence = result[0]
            max_confidence = max(confidence)
            predicted_index = np.argmax(confidence)
            predicted_category = CATEGORIES[predicted_index]

            # Update results count
            results_count[predicted_category] += 1
            print(f"Prediction: {predicted_category}, Confidence: {max_confidence * 100:.2f}%")
           # print(f"Counts: {results_count}")

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
sess.close()

print("Final counts:", results_count)

import requests

# Set the URL for your Flask API's /predict endpoint
url = 'http://127.0.0.1:5000/predict'

# Load the image you want to send
image_path = 'Samosa Test.jpg'  # Replace with your image file path

# Create a dictionary to hold the image file
files = {'image': open(image_path, 'rb')}

# Send the POST request
response = requests.post(url, files=files)

# Print the response
print(response.json())

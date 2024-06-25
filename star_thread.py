# import requests

# url = "http://127.0.0.1:8000/?tts=Postman is available for all the main operating"
# url1 = "http://127.0.0.1:8001/?tts=Postman is available for all the main operating"

# payload = {}
# headers = {
#   'accept': 'application/json'
# }

# response = requests.request("POST", url, headers=headers, data=payload)
# print(response.text)
# response = requests.request("POST", url1, headers=headers, data=payload)
# print(response.text)


import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Define the URL and headers
urls = [
    "http://127.0.0.1:8000/?tts=Postman is available for all the main operating",
    "http://127.0.0.1:8001/?tts=Another request for testing threading pool"
]
headers = {'accept': 'application/json'}

# Function to send a request
def send_request(url):
    response = requests.request("POST", url, headers=headers, data={})
    return response.text

# Function to manage sending requests with a delay
def send_requests_with_delay(urls, delay):
    with ThreadPoolExecutor(max_workers=len(urls)) as executor:
        futures = [executor.submit(send_request, url) for url in urls]
        
        for future in as_completed(futures):
            print(future.result())
            time.sleep(delay)  # Delay between processing each request

# Call the function with a 1-second delay between requests
send_requests_with_delay(urls, 1)

print("All requests have been processed.")

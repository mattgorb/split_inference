import requests
import json
import time

class ModelClient:
    def __init__(self, server_url):
        self.url = f"http://{server_url}/generate"
    
    def generate(self, prompt, max_length=100, temperature=0.7):
        data = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature
        }
        
        response = requests.post(self.url, json=data)
        return response.json()

if __name__ == "__main__":
    # Use the IP from terraform output
    client = ModelClient("0.0.0.0:8000")  # Replace with actual IP

    input_texts = [
                "Hello, how are you today?", 
                "What is the weather like?",
                "I like traveling by train because",
                "Artificial Intelligence (AI) has seen tremendous growth over the last decade, leading to advancements in natural language processing, computer vision, and robotics. What do you think the future of AI holds?", 
            ]
    
    for prompt in input_texts:
        # Test the service
        start_time = time.perf_counter()
        print(prompt)
        result = client.generate(prompt)
        print('here?')
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        

        print(f"\nGenerated text: {result}")
        print(f"API request took {elapsed_time:.4f} seconds.")

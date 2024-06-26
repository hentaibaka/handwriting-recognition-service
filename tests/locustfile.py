from locust import HttpUser, task, between, TaskSet, events
import os

class UserBehavior(TaskSet):
    @task
    def recognize(self):
        image_path = '/mnt/images/1.jpg'
        url = '/api/recognize/'

        if not os.path.exists(image_path):
            print(f"file not found {image_path}")
            return
        
        with open(image_path, 'rb') as image_file:
            files = {'image': image_file}
            with self.client.post(url, files=files, catch_response=True) as response:
                if response.status_code == 200:
                    try:
                        json_response = response.json()
                        response.success()
                    except ValueError:
                        response.failure("Response could not be decoded as json")
                else:
                    response.failure(f"Response failure with status {response.status_code}")

class MyUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 5)
    
@events.request.add_listener
def request_success(request_type, name, response_time, response_length, **kwargs):
    print(f"Request success: {request_type} {name} Response time: {response_time}ms Response length: {response_length} bytes")

@events.request.add_listener
def request_failure(request_type, name, response_time, response_length, exception, **kwargs):
    print(f"Request failure: {request_type} {name} Response time: {response_time}ms Response length: {response_length} bytes Exception: {exception}")

        
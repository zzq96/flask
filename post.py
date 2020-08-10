import requests
import os
PyTorch_REST_API_URL = 'http://127.0.0.1:5000/predict'
time_all = 0
def predict_result(image_path):
    global time_all
    # Initialize image path
    image = open(image_path, 'rb').read()
    payload = {'image': image}

    # Submit the request.
    time_start = time.time()
    r = requests.post(PyTorch_REST_API_URL, files=payload).json()
    time_all += time.time() - time_start
    # Ensure the request was successful.
    if r['success']:
        # Loop over the predictions and display them.
        pass
        # for (i, result) in enumerate(r['predictions']):
        #     print('{}. {}: {:.4f}'.format(i + 1, result['label'],
        #                                   result['probability']))
    # Otherwise, the request failed.
        print('Request success')
    else:
        print('Request failed')


import time
img_dir = r'D:\Code\python\BoxData\train'
cnt = 0
for img_file in os.listdir(img_dir):
    if '.png' not in img_file:
        continue
    cnt+=1
    img_file = os.path.join(img_dir, img_file)
    predict_result(img_file)

print(time_all / cnt)
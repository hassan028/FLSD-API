import os
import argparse

from helper import bytes_to_img
from syndicai import PythonPredictor


sample_data = "https://img.medscapestatic.com/pi/meds/ckb/45/37145tn.jpg"
output_dir = "./output"
save_response = True

def run(opt):
    sample_json = {"url": opt.image}

    model = PythonPredictor([])
    response = model.predict(sample_json)

    if opt.save:
        img, format = bytes_to_img(response)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        img_name = f"output.{format}"
        img.save(os.path.join(output_dir, img_name))

    if opt.response:
        print(response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default=sample_data, type=str, help='URL to a sample input data')
    parser.add_argument('--save', action='store_true', help='Save output image in the ./output directory')
    parser.add_argument('--response', default=True, type=bool, help='Print a response in the terminal')
    opt = parser.parse_args()
    run(opt)
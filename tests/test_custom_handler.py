import pytest
import torch
import json
from handlers.custom_handler import CustomHandler


@pytest.fixture
def handler():
    resnet34 = torch.jit.load('resnet34.pt')
    with open('index_to_name.json') as f:
        index_to_name = json.load(f)

    handler = CustomHandler()
    handler.model = resnet34
    handler.mapping = index_to_name
    return handler


@pytest.fixture
def example_img():
    data = open('kitten_small.jpg', 'rb').read()
    return data


def invoke(handler, data):
    data = handler.preprocess(data)
    data = handler.inference(data)
    data = handler.postprocess(data)
    return data


def test_handler(handler, example_img):
    req = Req(data=example_img)
    data = invoke(handler, req)
    assert(data[0]['label'] == 'tabby')


class Req():
    def __init__(self, data):
        self.data = data

    def get(self, _something):
        return self.data

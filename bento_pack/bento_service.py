import bentoml
from typing import List
from torchvision import transforms
from src.models.net import Net
from bentoml.adapters import ImageInput
from bentoml.frameworks.pytorch import PytorchModelArtifact
from prometheus_client import Summary, Gauge, Counter
import imageio
import torch
import torch.nn.functional as f

REQUEST_TIME = Summary(name='predict_request_processing_time',
                       documentation='Time spend predict processing request',
                       namespace='BENTOML')

probs_gauge = Gauge(name="predict_probs_rate",
                    documentation='predict probs rate',
                    labelnames=['class'],
                    namespace='BENTOML')

FASHION_MNIST_CLASSES = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


@bentoml.env(pip_packages=["torch", "torchvision", "imageio==2.10.3"])
@bentoml.artifacts([PytorchModelArtifact('model')])
class PytorchModelService(bentoml.BentoService):

    @bentoml.utils.cached_property  # reuse transformer
    def transform(self):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.Normalize((0.1307,), (0.3081,))])

    @REQUEST_TIME.time()
    @bentoml.api(input=ImageInput(), batch=True)
    def predict(self, image_arrays: List[imageio.core.util.Array]) -> List[str]:
        result = []
        for x in image_arrays:
            x = self.transform(x)
            outputs = self.artifacts.model(x)
            _, output_classes = outputs.max(dim=1)
            probs = torch.max(f.softmax(outputs))
            probs_gauge.labels(output_classes.item()).set(float(probs.item()))
            result.append({"probs": "{:.1%}".format(probs.item()), "classes": FASHION_MNIST_CLASSES[output_classes.item()]})
        return result

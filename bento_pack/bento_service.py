import bentoml
from torchvision import transforms
from net import Net
from bentoml.adapters import ImageInput
from bentoml.frameworks.pytorch import PytorchModelArtifact
from prometheus_client import Summary
import torch
import torch.nn.functional as f

REQUEST_TIME = Summary(name='predict_request_processing_time',
                       documentation='Time spend predict processing request',
                       namespace='BENTOML')


@bentoml.env(pip_packages=["torch", "torchvision", "imageio==2.10.3"])
@bentoml.artifacts([PytorchModelArtifact('model')])
class PytorchModelService(bentoml.BentoService):

    @bentoml.utils.cached_property  # reuse transformer
    def transform(self):
        return transforms.Compose([transforms.ToTensor(),
                                   transforms.Grayscale(num_output_channels=1),
                                   transforms.Normalize((0.1307,), (0.3081,))])

    @REQUEST_TIME.time()
    @bentoml.api(input=ImageInput(), batch=False)
    def predict(self, img):
        x = self.transform(img)
        outputs = self.artifacts.model(x)
        _, output_classes = outputs.max(dim=1)
        probs = torch.max(f.softmax(outputs))
        return {"probs": "{:.1%}".format(probs.item()),
                "output_classes": output_classes.item()}

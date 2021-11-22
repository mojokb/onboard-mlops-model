import bentoml
from net import Net
from bentoml.adapters import ImageInput
from bentoml.frameworks.pytorch import PytorchModelArtifact
from prometheus_client import Summary

REQUEST_TIME = Summary(name='predict_request_processing_time', documentation='Time spend predict processing request', namespace='PREFIX')

@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PytorchModelArtifact('model')])
class PytorchModelService(bentoml.BentoService):
    @REQUEST_TIME.time()
    @bentoml.api(input=ImageInput(), batch=True)
    def predict(self, imgs):
        outputs = self.artifacts.model(imgs)
        return outputs

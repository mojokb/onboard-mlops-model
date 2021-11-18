import bentoml
from bentoml.adapters import ImageInput
from bentoml.frameworks.pytorch import PytorchModelArtifact


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([PytorchModelArtifact('model')])
class PytorchModelService(bentoml.BentoService):
    @bentoml.api(input=ImageInput(), batch=True)
    def predict(self, imgs):
        outputs = self.artifacts.net(imgs)
        return outputs
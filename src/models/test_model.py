import torch
from torchvision import transforms
from itertools import chain
from sklearn.metrics import f1_score
from src.models.onboard_dataset import OnboardDataset


class TestModel:
    def __init__(self, model_path="/workdir/models/model.pt", 
                 dataset_path="/workdir/data/processed/test_set.npz"):
        self.classes = 10
        torch.manual_seed(1)
        self.device = torch.device("cpu")
        self.model_path = model_path
        self.dataset_path = dataset_path
        self._load_model()

    def _load_model(self):
        self.model = torch.load(self.model_path)

    def test(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        test_batch_size = 256
        test_loader = torch.utils.data.DataLoader(
            OnboardDataset(path=self.dataset_path,
                           transform=transform),
            batch_size=test_batch_size, shuffle=True)

        self.model.eval()
        predictions_list = []
        labels_list = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                labels_list.append(target)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                predictions_list.append(pred)

        predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
        labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
        predictions_l = list(chain.from_iterable(predictions_l))
        labels_l = list(chain.from_iterable(labels_l))

        ap_avg = 0.0
        for i in range(self.classes):
            ap_avg += f1_score(labels_l, predictions_l, labels=[i], average='weighted')
        return ap_avg / self.classes


if __name__ == "__main__":
    test_model = TestModel(model_path="/Users/kb/PycharmProjects/mlflow-test/model.pt",
                           dataset_path="/Users/kb/PycharmProjects/mlflow-test/train_set2.npz")
    print(test_model.test())


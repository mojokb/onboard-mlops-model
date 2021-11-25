import torch
from torchvision import transforms
from itertools import chain
from sklearn.metrics import f1_score
from src.models.onboard_dataset import OnboardDataset


class TestModel:
    def __init__(self, workdir_base='/workdir'):
        self.classes = 10
        no_cuda = True
        use_cuda = not no_cuda and torch.cuda.is_available()
        torch.manual_seed(1)
        self.device = torch.device("cuda" if use_cuda else "cpu")
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

        mnist_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_batch_size = 1000
        self.test_loader = torch.utils.data.DataLoader(
            OnboardDataset(path=f'{workdir_base}/data/processed/test_set.npz', 
                           transform=mnist_transform),
            batch_size=test_batch_size, shuffle=True, **kwargs)

    def load_model(self, path="/workdir/models/model.pt"):
        self.model = torch.load(path)

    def test(self):
        self.model.eval()
        predictions_list = []
        labels_list = []
        with torch.no_grad():
            for data, target in self.test_loader:
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
    test_model = TestModel(testset_base=".")
    test_model.load_model("./model/model.pt")
    print(test_model.test())


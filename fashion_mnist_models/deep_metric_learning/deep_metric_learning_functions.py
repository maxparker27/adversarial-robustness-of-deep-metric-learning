import torch
from pytorch_metric_learning import losses, reducers, miners, distances, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator


def train(dataloader, model, loss_fn, optimizer, miner, device):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        outputs = model(X)
        hard_pairs = miner(outputs, y)

        loss = loss_fn(outputs, y, hard_pairs)

        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(test_embeddings,
                                                  train_embeddings,
                                                  test_labels,
                                                  train_labels,
                                                  False)
    print(
        "Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))


# def testDML(dataloader, model, loss_fn, device):
#     size = len(dataloader.dataset)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= size
#     correct /= size
#     print(
#         f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

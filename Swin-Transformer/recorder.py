import torch


class recorder:
    def __init__(self, writer, split='train'):
        self.writer = writer
        self.split = split
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.loss_count = 0
        self.epoch_count = 0

    def update_states(self, outputs, labels):
        # print(outputs.shape)
        # print(labels.shape)
        # print('--------')
        predictions = outputs.argmax(dim=-1)
        labels = labels.argmax(dim=-1)
        # print(predictions.shape)
        # print(labels.shape)
        # print('-------')
        self.TP += torch.where((predictions == labels.cuda()) & (predictions == 1), 1, 0).sum()
        self.TN += torch.where((predictions == labels.cuda()) & (predictions == 0), 1, 0).sum()
        self.FP += torch.where((predictions != labels.cuda()) & (predictions == 1), 1, 0).sum()
        self.FN += torch.where((predictions != labels.cuda()) & (predictions == 0), 1, 0).sum()

    def cal_metrics(self):
        accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        recall = self.TP / (self.TP + self.FN)
        precision = self.TP / (self.FP + self.TP)
        f1_score = 2 * (recall * precision) / (recall + precision)

        self.writer.add_scalar("Accuracy/"+self.split, accuracy, self.epoch_count)
        self.writer.add_scalar("Recall/"+self.split, recall, self.epoch_count)
        self.writer.add_scalar("Precision/"+self.split, precision, self.epoch_count)
        self.writer.add_scalar("f1/"+self.split, f1_score, self.epoch_count)
        self.writer.add_scalars("Confusion//"+self.split, {'TP': self.TP,
                                                  'TN': self.TN,
                                                  'FP': self.FP,
                                                  'FN': self.FN},
                                self.epoch_count)
        self.TP, self.TN, self.FP, self.FN = 0, 0, 0, 0
        self.epoch_count += 1

    def write_loss(self, loss):
        self.writer.add_scalar("loss/"+self.split, loss, self.loss_count)
        self.loss_count += 1

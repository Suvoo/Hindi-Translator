import torch

n_epochs = 50
avg_accuracy = 0
total_accuracy = 0
test_loss = 0
accuracy = 0
dhcd_model.eval()

for epoch in range(n_epochs):
    
    for img, label in test_loader:
        if train_on_gpu:
            img = img.cuda()
            label = label.cuda()
        predicted_label = dhcd_model(img)
        loss = criterion(predicted_label, label)
        test_loss = test_loss + loss.item()

        top_probab, top_label = predicted_label.topk(1, dim=1)
        equals = top_label == label.view(*top_label.shape)
        accuracy = accuracy + torch.mean(equals.type(torch.FloatTensor))

    test_loss = test_loss/len(test_loader)
    accuracy = accuracy/len(test_loader)
    total_accuracy = total_accuracy + accuracy

    print("Epoch: {} Test Loss: {} Accuracy: {}".format(epoch+1, test_loss, accuracy))

avg_accuracy = total_accuracy/(n_epochs) * 100
print("______________________\nAverage Accuracy: {:.3f}%\n______________________".format(avg_accuracy))
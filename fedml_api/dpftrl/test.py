#define the network in pytorch (with Z)
import torch.nn as nn
import torch
from torchvision import datasets, transforms

from dp_ftrl import FTRLM

class NeuralNetworkZ(nn.Module):
    def __init__(self, input_dim, hidden_dim, lr_w, lr_z, batch_size):
        super(NeuralNetworkZ, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim[0])
        self.lin2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.z2 = nn.Parameter(torch.zeros(batch_size, hidden_dim[1]), requires_grad=True)
        self.lin3 = nn.Linear(hidden_dim[1], hidden_dim[2])

        self.apply(self.init_weights_biases)
        self.apply(self.init_z)

        self.optimizer_w = torch.optim.Adam(self.get_w(), lr=lr_w)
        self.optimizer_z = torch.optim.Adam(self.get_z(), lr=lr_z)

        self.criterion_z = nn.MSELoss()

    def init_weights_biases(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight, gain=1.0) #modify with custom function later
            nn.init.normal_(m.bias, mean=0.0, std=1.0)

    def get_w(self):
        return [self.input_layer.weight, self.lin2.weight, self.lin3.weight]

    def init_z(self, m):
        list_z = self.get_z()
        for z in list_z:
            z = nn.init.xavier_normal_(z, gain=1.0)

    def get_z(self):
        return [self.z2]


    def forward(self, x, training=True):
        if training:
            x = self.input_layer(x)
            x = nn.Tanh()(x)
            z2_tar = nn.Tanh()(self.lin2(x))
            logits = self.lin3(self.z2)
            return logits, z2_tar

        else:
            x = self.input_layer(x)
            x = nn.Tanh()(x)
            x = self.lin2(x)
            x = nn.Tanh()(x)
            x = self.lin3(x)
            return x


    def z_loss(self, x):
        logits, z2_tar = self.forward(x)
        loss_z2 = self.criterion_z(self.z2, z2_tar)

        return logits, loss_z2

    def compute_ce_loss(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss()
        final_loss = ce_loss(logits, targets)
        return final_loss


    def loss(self, x, targets, requires_z=True):
        if requires_z:
            logits, loss_z2 = self.z_loss(x)
            final_loss = self.compute_ce_loss(logits, targets)
            return final_loss, loss_z2

        else:
            logits = self.forward(x, requires_z=False)
            final_loss = self.compute_ce_loss(logits, targets)
            return final_loss
        

if __name__ == '__main__':
    input_dim = 784

    #Parameters setup
    classes = 10
    lr_w = 0.001
    lr_z = 0.1
    batch_size = 100
    hidden_dim = [100, 100, 10]
    epochs = 3

    #Get MNIST data and define transforms
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])
    trainset = datasets.MNIST('mnist_train', download=True, train=True, transform=transform)
    testset = datasets.MNIST('mnist_test', download=True, train=False, transform=transform)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)

    model = NeuralNetworkZ(input_dim, hidden_dim, lr_w, lr_z, batch_size)

    bs = 1000
    noise_multiplier = 7.0
    print("#############params###############")
    param_list = []
    param_shapes = [p.shape for p in model.parameters()]
    for p in model.parameters():
        param_list.append(p)
    print("#############params###############")


    optimizer_FTRLM = FTRLM( # noqa
            params=param_list,
            model_param_sizes=param_shapes,
            lr=0.01,
            momentum=0.9,
            nesterov=True,
            noise_std=(noise_multiplier / bs),
            max_grad_norm=1.0,
            seed=0,
            efficient=True,
            )
    
    model.optimizer_w = optimizer_FTRLM

    criterion = nn.CrossEntropyLoss()

    device = 'cpu'
    n_total_steps = len(trainloader)

    beta = 10
    z_iter = 100

    for epoch in range(epochs):
        #reinit Z here every epoch

        for i, (images, labels) in enumerate(trainloader):
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            # Forward pass
            #outputs = model(images)


            for e in range(z_iter):
                #progressively increase beta/gamma
                L, loss_z2 = model.loss(images, labels)
                loss_z2_temp = beta*loss_z2

                total_loss_temp = L + loss_z2_temp

                model.optimizer_z.zero_grad()
                total_loss_temp.backward()
                #loss_z2_temp.backward(retain_graph=True)

                model.optimizer_z.step()
                #scheduler.step(total_loss_temp) #if scheduler is used

            loss, loss_z2  = model.loss(images, labels)
            total_loss = loss + beta*loss_z2
            # Backward and optimize
            model.optimizer_w.zero_grad()
            #loss.backward()
            total_loss.backward()
            model.optimizer_w.step()

            if (i+1) % 100 == 0:
                print (f'Epoch [{epoch+1}/{epochs}], Step[{i+1}/{n_total_steps}], Loss: {total_loss.item():.8f}, Loss Z2: {loss_z2.item():.10f}, CE Loss: {loss.item():.10f}')

                with torch.no_grad():
                    n_correct = 0
                    n_samples = 0
                    for images, labels in testloader:
                        images = images.reshape(-1, 28*28).to(device)
                        labels = labels.to(device)
                        outputs = model(images, training=False)
                        # max returns (value ,index)
                        _, predicted = torch.max(outputs.data, 1)
                        n_samples += labels.size(0)
                        n_correct += (predicted == labels).sum().item()

                acc = 100.0 * n_correct / n_samples
                print(f'Accuracy of the network on the 10000 test images: {acc} %')

    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for images, labels in testloader:
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)
            outputs = model(images, training=False)
            # max returns (value ,index)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
from types import SimpleNamespace
from typing import Any

import numpy as np

import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
import torchvision

from norse.torch.models.conv import ConvNet4
from norse.torch.module.encode import ConstantCurrentLIFEncoder


class LIFConvNet(torch.nn.Module):
	def __init__(
		self,
		input_features,
		seq_length,
		input_scale,
		model="super",
	):
		super(LIFConvNet, self).__init__()
		self.constant_current_encoder = ConstantCurrentLIFEncoder(seq_length=seq_length)
		self.input_features = input_features
		self.rsnn = ConvNet4(method=model)
		self.seq_length = seq_length
		self.input_scale = input_scale

	def forward(self, x):
		batch_size = x.shape[0]
		x = self.constant_current_encoder(
			x.view(-1, self.input_features) * self.input_scale
		)

		x = x.reshape(self.seq_length, batch_size, 1, 28, 28)
		voltages = self.rsnn(x)
		m, _ = torch.max(voltages, 0)
		log_p_y = torch.nn.functional.log_softmax(m, dim=1)
		return log_p_y


def train(
	model,
	device,
	train_loader,
	optimizer,
	epoch,
	clip_grad,
	grad_clip_value,
	epochs,
	log_interval,
	writer,
):
	model.train()
	losses = []

	batch_len = len(train_loader)
	step = batch_len * epoch

	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = torch.nn.functional.nll_loss(output, target)
		loss.backward()

		if clip_grad:
			torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)

		optimizer.step()
		step += 1

		if batch_idx % log_interval == 0:
			print(
				"Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
					epoch,
					epochs,
					batch_idx * len(data),
					len(train_loader.dataset),
					100.0 * batch_idx / len(train_loader),
					loss.item(),
				)
			)

		if step % log_interval == 0:
			_, argmax = torch.max(output, 1)
			accuracy = (target == argmax.squeeze()).float().mean()
			writer.add_scalar("Loss/train", loss.item(), step)
			writer.add_scalar("Accuracy/train", accuracy.item(), step)

			for tag, value in model.named_parameters():
				tag = tag.replace(".", "/")
				writer.add_histogram(tag, value.data.cpu().numpy(), step)
				if value.grad is not None:
					writer.add_histogram(
						tag + "/grad", value.grad.data.cpu().numpy(), step
					)

	mean_loss = np.mean(losses)
	return losses, mean_loss


def test(model, device, test_loader, epoch, method, writer):
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += torch.nn.functional.nll_loss(
				output, target, reduction="sum"
			).item()  # sum up batch loss
			pred = output.argmax(
				dim=1, keepdim=True
			)  # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	test_loss /= len(test_loader.dataset)

	accuracy = 100.0 * correct / len(test_loader.dataset)
	print(
		f"\nTest set {method}: Average loss: {test_loss:.4f}, \
			Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.0f}%)\n"
	)
	writer.add_scalar("Loss/test", test_loss, epoch)
	writer.add_scalar("Accuracy/test", accuracy, epoch)

	return test_loss, accuracy


def save(path, epoch, model, optimizer, is_best=False):
	torch.save(
		{
			"epoch": epoch + 1,
			"model_state_dict": model.state_dict(),
			"optimizer_state_dict": optimizer.state_dict(),
			"is_best": is_best,
		},
		path,
	)


def load(path, model, optimizer):
	checkpoint = torch.load(path)
	model.load_state_dict(checkpoint["model_state_dict"])
	optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
	model.train()
	return model, optimizer


def main(args: Any) -> None:
	writer = SummaryWriter()

	torch.manual_seed(args.random_seed)
	np.random.seed(args.random_seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(args.random_seed)
		torch.backends.cudnn.enabled = True
		torch.backends.cudnn.benchmark = True

	device = torch.device(args.device)

	image_transform = torchvision.transforms.Compose([
		torchvision.transforms.Grayscale(),
		torchvision.transforms.Resize((28, 28)),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize((0.1307,), (0.3081,))])
	kwargs = {"num_workers": 1, "pin_memory": True} if args.device == "cuda" else {}
	train_loader = torch.utils.data.DataLoader(
		torchvision.datasets.SVHN("SVHN", "train", image_transform, download = True),
		args.batch_size,
		True,
		**kwargs)
	test_loader = torch.utils.data.DataLoader(
		torchvision.datasets.SVHN("SVHN", "test", image_transform, download = True),
		args.batch_size,
		**kwargs)

	input_features = 28 * 28

	model = LIFConvNet(
		input_features,
		args.seq_length,
		input_scale = args.input_scale,
		model = args.method,
	).to(device)

	if args.optimizer == "sgd":
		optimizer = torch.optim.SGD(model.parameters(), lr = args.learning_rate)
	elif args.optimizer == "adam":
		optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate)

	if args.only_output:
		optimizer = torch.optim.Adam(model.out.parameters(), lr = args.learning_rate)

	training_losses = []
	mean_losses = []
	test_losses = []
	accuracies = []

	for epoch in range(args.epochs):
		training_loss, mean_loss = train(
			model,
			device,
			train_loader,
			optimizer,
			epoch,
			clip_grad = args.clip_grad,
			grad_clip_value = args.grad_clip_value,
			epochs = args.epochs,
			log_interval = args.log_interval,
			writer = writer,
		)
		test_loss, accuracy = test(
			model, device, test_loader, epoch, method = args.method, writer = writer
		)

		training_losses += training_loss
		mean_losses.append(mean_loss)
		test_losses.append(test_loss)
		accuracies.append(accuracy)

		max_accuracy = np.max(np.array(accuracies))

		if (epoch % args.model_save_interval == 0) and args.save_model:
			model_path = f"mnist-{epoch}.pt"
			save(
				model_path,
				model = model,
				optimizer = optimizer,
				epoch = epoch,
				is_best = accuracy > max_accuracy,
			)

	np.save("training_losses.npy", np.array(training_losses))
	np.save("mean_losses.npy", np.array(mean_losses))
	np.save("test_losses.npy", np.array(test_losses))
	np.save("accuracies.npy", np.array(accuracies))
	model_path = "mnist-final.pt"
	save(
		model_path,
		epoch = epoch,
		model = model,
		optimizer = optimizer,
		is_best = accuracy > max_accuracy,
	)


if __name__ == "__main__":
	args = SimpleNamespace()
	args.save_grads = False
	args.grad_save_interval = 10
	args.refrac = False
	args.input_scale = 1.0
	args.find_learning_rate = False
	args.device = "cuda" if torch.cuda.is_available() else "cpu"
	args.epochs = 10
	args.seq_length = 200
	args.batch_size = 32
	args.method = "super"
	args.prefix = ""
	args.optimizer = "adam"
	args.clip_grad = False
	args.grad_clip_value = 1.0
	args.learning_rate = 2e-3
	args.log_interval = 10
	args.model_save_interval = 50
	args.save_model = True
	args.big_net = False
	args.only_output = False
	args.random_seed = 1234

	main(args)

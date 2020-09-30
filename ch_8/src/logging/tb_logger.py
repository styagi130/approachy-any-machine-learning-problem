from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger(SummaryWriter):
	def __init__(self, logdir):
		super(TensorboardLogger, self).__init__(logdir)

	def log_training(self, reduced_loss, grad_norm, iteration):
			self.add_scalar("training_step.loss", reduced_loss, iteration)
			self.add_scalar("training_step.grad_norm", grad_norm, iteration)


	def log_training_grad_norm(self, reduced_loss, grad_norm, iteration):
			self.add_scalar("training_epoch.loss", reduced_loss, iteration)
			self.add_scalar("training_epoch.grad_norm", grad_norm, iteration)


	def log_validation(self, reduced_loss,accuracy, model, iteration):
		self.add_scalar("validation.loss", reduced_loss, iteration)

import pytorch_lightning as pl

class BaseTask(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Set the task name to the name of the current class
        self.hparams["task_name"] = type(self).__name__
        # Call the post initialization method
        self.post_init()

    def post_init(self):
        """
        This method is called at the end of the BaseTask __init__ method.
        """
        pass

    def info(self):
        """
        Returns the task description defined in the child class.
        """
        # Check if the child class has a DESCRIPTION attribute
        if hasattr(self, 'DESCRIPTION'):
            return self.DESCRIPTION
        else:
            # Return a default message or raise an error if DESCRIPTION is not defined
            return "No description provided."


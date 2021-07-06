dl_mnist.py contains the script to run, train and test the model. This will also implement the chosen adversarial attack
as well give accuracies. It is dependent on the other files in the directory for the model definition and various functions
to do with training and testing the model as well as performing the adversarial attacks.

deep_learning_functions.py contains the the training and testing functions for the CNN model.

adversarial_attacks.py contains the functions necessary to implement the chosen adversarial attacks, as well as test the
accuracy of the model on the various adversarial examples.

model_class_mnist.py contains the PyTorch model class. It defines all of the layers in the network as well as components such
as the choice of transfer function.

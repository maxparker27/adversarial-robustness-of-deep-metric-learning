import foolbox as fb
from foolbox import accuracy
import torch


def determine_attack(which_attack):
    if which_attack == "FGSM":
        return fb.attacks.LinfFastGradientAttack()
    elif which_attack == "CarliniWagner":
        return fb.attacks.L2CarliniWagnerAttack()
    elif which_attack == "PGD":
        return fb.attacks.LinfProjectedGradientDescentAttack()


def perform_attack(choose_attack, model, epsilons, images, labels):

    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    attack = determine_attack(choose_attack)
    print("\nUsing {} attack:".format(choose_attack))

    for epsilon in epsilons:
        raw_advs, clipped_advs, success = attack(
            fmodel, images, labels, epsilons=epsilon)

        success = success.type(torch.FloatTensor)

        clean_acc = accuracy(fmodel, images, labels)
        print(f"clean accuracy:  {clean_acc * 100:.1f} %")

        robust_accuracy = 1 - success.mean(axis=-1)
        print("robust accuracy for perturbations with epsilon = {} is {}%.".format(
            epsilon, round(robust_accuracy.item() * 100, 4)))
    print("{} attack is complete.".format(choose_attack))

import pickle

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


LABELS = ['front', 'back', 'fronttwo', 'mid', 'full', 'None']


def main():
    with open('../green_diffmodel_uncluttered_table_project_room_black.pkl', 'rb') as f:
        guesses_for_class = pickle.load(f)

    correct_labels = []
    guesses = []

    correct = 0
    incorrect = 0

    for correct_label, guess_counts in guesses_for_class.items():
        for guess, count in guess_counts.items():
            for i in range(count):
                correct_labels.append(correct_label)
                guesses.append(guess)

                if correct_label == guess:
                    correct += 1
                else:
                    incorrect += 1

    cm = confusion_matrix(correct_labels, guesses, labels=LABELS)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot()
    plt.show()

    print('correct', correct)
    print('incorrect', incorrect)
    print(correct/(correct + incorrect))


if __name__ == '__main__':
    main()
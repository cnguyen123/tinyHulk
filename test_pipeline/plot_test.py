import pickle

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


LABELS = ['full', 'front', 'fronttwo', 'back', 'mid']

data_file = "../data/statistic_data/full_pipeline_test/fromserver/meccano_bike/green_diff/meccano_bike_frcnn_guess.pkl"
plot_file = "../data/statistic_data/full_pipeline_test/graph/meccano_bike/real/realmd_result.png"
def main():
    with open(data_file, 'rb') as f:
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
    #disp.plot()
    #plt.show()
    #plt.savefig(plot_file)

    print('correct', correct)
    print('incorrect', incorrect)
    print('Accuracy', correct/(corect + incorrect))


if __name__ == '__main__':
    main()

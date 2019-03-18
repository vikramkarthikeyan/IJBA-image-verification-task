import pandas as pd

from sklearn import metrics
import matplotlib.pyplot as plt

def plot_ROC(fpr, tpr, roc_auc, split_number):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
            lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for split:' +str(split_number))
    plt.legend(loc="lower right")
    plt.savefig('./ROC/split_' + str(split_number) + '.png')

df = pd.read_csv('./ROC/split_' + str(1) + '.csv')

for i in range(2, 11):
    temp_df = pd.read_csv('./ROC/split_' + str(i) + '.csv')
    df = df.append(temp_df)

# Compute ROC for data from all the splits
actual_scores = df['actual']
similarity_scores = df['cosine']
fpr, tpr, thresholds = metrics.roc_curve(actual_scores, similarity_scores, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

TAR = {
    0.001: '',
    0.01: '',
    0.1: '',
    0.2: ''
}

for i, FAR in enumerate(fpr):
    if round(FAR,4) == 0.001:
        TAR[0.001] = tpr[i]
    if round(FAR, 2) == 0.01:
        TAR[0.01] = tpr[i]
    if round(FAR, 1) == 0.1:
        TAR[0.1] = tpr[i]
    if round(FAR, 1) == 0.2:
        TAR[0.2] = tpr[i]


print(TAR)




# Plot the ROC curve
plot_ROC(fpr, tpr, roc_auc, 'consolidated')
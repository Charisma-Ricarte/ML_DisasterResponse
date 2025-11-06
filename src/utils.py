import matplotlib.pyplot as plt

def plot_label_distribution(df, label_col):
    df[label_col].value_counts().plot(kind="bar")
    plt.title("Label Distribution")
    plt.show()

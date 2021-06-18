import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing

def screen_data(petal_sepal, data):
    data_select = data.copy()
    if petal_sepal == "sepal":
        data_select = data.iloc[:, :-3]
        return data_select
    elif petal_sepal == "petal":
        data_select = data.iloc[:, 2:-1]
        return data_select

def scater_plot(data, value, size):
    x_values = f"{value}_length"
    y_values = f"{value}_width"
    x_label = f"Length of {value}"
    y_label = f"Width of {value}"
    fig, ax = plt.subplots(figsize=size)
    if "clusters" in data.columns:
        ax.scatter(data[x_values], data[y_values], c=data["clusters"], cmap="rainbow")
    else:
        ax.scatter(data[x_values], data[y_values])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return fig

def elbow_method(clusters, data, size):
    wcss = list()
    for c in range(1, clusters+1):
        kmeans = KMeans(c)
        kmeans.fit(data)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)
    fig, ax = plt.subplots(figsize=size)
    ax.plot(range(1, clusters+1), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster Sum of Squares')
    return fig

def cluster_data(number_of_clusters, data):
    data_c = data.copy()
    kmeans = KMeans(number_of_clusters)
    kmeans.fit(data_c)
    data_c["clusters"] = kmeans.fit_predict(data_c)
    return data_c

def scale_cluster_data(number_of_clusters, data):
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)

    data_c = data.copy()
    kmeans = KMeans(number_of_clusters)
    kmeans.fit(data_scaled)
    data_c["clusters"] = kmeans.fit_predict(data_scaled)
    return data_c
def scale_data(data):
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled

def screen_data(petal_sepal, data):
    if petal_sepal == "sepal":
        data_select = data.iloc[:, :-3]
        return data_select
    elif petal_sepal == "petal":
        data_select = data.iloc[:, 2:-1]
        return data_select

def scater_plot(data, value):
    x_values = f"{value}_length"
    y_values = f"{value}_width"
    x_label = f"Length of {value}"
    y_label = f"Width of {value}"

    if "clusters" in data.columns:
        plt.scatter(data[x_values], data[y_values], c=data["clusters"], cmap="rainbow")
    else:
        plt.scatter(data[x_values], data[y_values])
        
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    return plt.show()

def elbow_method(clusters, data):
    wcss = list()
    for c in range(1, clusters+1):
        kmeans = KMeans(c)
        kmeans.fit(data)
        wcss_iter = kmeans.inertia_
        wcss.append(wcss_iter)
    
    plt.plot(range(1, clusters+1), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-cluster Sum of Squares')
    return plt.show()

def cluster_data(number_of_clusters, data):
    data_c = data.copy()
    kmeans = KMeans(number_of_clusters)
    kmeans.fit(data_c)
    data_c["clusters"] = kmeans.fit_predict(data_c)
    return data_c
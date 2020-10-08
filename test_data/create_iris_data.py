import statsmodels.api as sm

def load_data():
    ###import iris data set, which is commonly used in R
    iris = sm.datasets.get_rdataset("iris").data
    iris_short_names = iris.rename(columns={'Sepal.Length': 'x1', 'Sepal.Width': 'x2', 'Petal.Length': 'x3', 'Petal.Width': 'x4', 'Species': "y"})
    return iris_short_names


if __name__ == '__main__':
    data = load_data()
    x = data.iloc[:, 0:4]
    y = data.iloc[:,4]
    x.to_csv("./x.csv",index=False)
    y.to_csv("./y.csv",index=False)
    print("dataset created")
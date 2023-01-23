import pandas as pd
import numpy as np
import math
import sklearn.datasets
import ipywidgets as widgets

##Seaborn for fancy plots. 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (15,10)
# plt.rcParams.update({'figure.autolayout': True})

class edaDF:
    """
    A class used to perform common EDA tasks

    ...

    Attributes
    ----------
    data : dataframe
        a dataframe on which the EDA will be performed
    target : str
        the name of the target column
    cat : list
        a list of the names of the categorical columns
    num : list
        a list of the names of the numerical columns
    outliers : dataframe
        a dataframe of outliers according to outlier bounds set
    outlierBounds: list
        a list of outlier conditions 

    Methods
    -------
    setCat(catList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        catlist : list
            The list of column names that are categorical

    setNum(numList)
        sets the cat variable listing the categorical column names to the list provided in the argument catList
        
        Parameters
        ----------
        numlist : list
            The list of column names that are numerical

    getNum(self)
        retuns a list of numerical features

    getCat(self)
        returns a list of categorical features

    countPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned.
    
    histPlots(self, splitTarg=False, show=True)
        generates countplots for the categorical variables in the dataset 

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the countplot to split the data by the target value
        show : bool
            If true, display the graphs when the function is called. Otherwise the figure is returned. 

    pairPlot(self, splitTarg=False, pairplot=False)
        generates pairplot for variables in the dataset

        Parameters
        ----------
        splitTarg : bool
            If true, use the hue function in the pairplot to split the data by the target value

    correlationMatrix(self)
        generates a correlationMatrix for all numerical features in the dataset

    setOutlierBounds(self, column, low=None, high=None, equals=None)
        sets outliers of one column in the data and adds the outliers to a dataframe. Bounds are exclusive of value specified in high and low arguments.

        Parameters
        ----------
        low : int or float
            if specified, sets lower bound for outliers
        high : int or float
            if specified, sets upper bound for outliers
        equals : int or float
            if specified, equivalent values are considered outliers

    showOutliers(self)
        returns dataframe of outliers specified by setOutlierBounds()

    displayOutlierBounds(self)
        prints list of outlier conditions

    targetBalance(self)
        returns dataframe showing the number of entries in each category in the target

    checkNulls(self)
        returns dataframe showing the number of null values in each column of the dataset

    fullEDA(self, pairplot=False)
        Displays the full EDA process. 

        Parameters
        ----------
        pairplot : bool
            If true, pairplot will be included in the full EDA. Otherwise, no figure will be shown
    """
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cat = []
        self.num = []
        self.outliers = pd.DataFrame()
        self.outlierBounds = []

    def info(self):
        return self.data.info()

    def describe(self):
        return self.data.describe().T

    def giveTarget(self):
        return self.target
        
    def setCat(self, catList):
        self.cat = catList
    
    def setNum(self, numList):
        self.num = numList

    def getNum(self):
        return self.num

    def getCat(self):
        return self.cat

    def countPlots(self, splitTarg=False, show=True):
        if len(self.cat) > 0:
            n = len(self.cat)
            cols = 3
            figure, ax = plt.subplots(math.ceil(n/cols), cols)
            r = 0
            c = 0
            for col in self.cat:
                if splitTarg == False:
                    sns.countplot(data=self.data, x=col, ax=ax[r][c])
                if splitTarg == True:
                    sns.countplot(data=self.data, x=col, hue=self.target, ax=ax[r][c])
                c += 1
                if c == cols:
                    r += 1
                    c = 0
            figure.set_tight_layout(True)
            if show == True:
                figure.show()
            return figure

    def histPlots(self, kde=True, splitTarg=False, show=True):
        if len(self.num) > 0:
            n = len(self.num)
            cols = 3
            figure, ax = plt.subplots(math.ceil(n/cols), cols)
            r = 0
            c = 0
            for col in self.num:
                #print("r:",r,"c:",c)
                if splitTarg == False:
                    sns.histplot(data=self.data, x=col, kde=kde, ax=ax[r][c])
                if splitTarg == True:
                    sns.histplot(data=self.data, x=col, hue=self.target, kde=kde, ax=ax[r][c])
                c += 1
                if c == cols:
                    r += 1
                    c = 0
            figure.set_tight_layout(True)
            if show == True:
                figure.show()
            return figure

    def pairPlot(self, splitTarg=False):
        if splitTarg == True:
            sns.pairplot(self.data, hue=self.target)
        if splitTarg == False:
            sns.pairplot(self.data)
        plt.show()

    def correlationMatrix(self):
        if len(self.num) > 0:
            figure, ax = plt.subplots()
            cTable = self.data[self.num].apply(pd.to_numeric, errors='coerce')
            cTable = cTable.corr()
            mask = np.triu(np.ones_like(cTable, dtype=bool))
            sns.heatmap(cTable, center=0, linewidths=.5, annot=True, cmap="YlGnBu", yticklabels=True, mask=mask)
            plt.show()

    def setOutlierBounds(self, column, low=None, high=None, equals=None):
        if low is not None and high is not None:
            df_outlier = self.data[(self.data[column] < low) | (self.data[column] > high)]
            self.outlierBounds.append(f"{column} > {high}")
            self.outlierBounds.append(f"{column} < {low}")
        elif high is not None:
            df_outlier = self.data[self.data[column] > high]
            self.outlierBounds.append(f"{column} > {high}")
        elif low is not None:
            df_outlier = self.data[self.data[column] < low]
            self.outlierBounds.append(f"{column} < {low}")
        elif equals is not None:
            df_outlier = self.data[self.data[column] == equals]
            self.outlierBounds.append(f"{column} == {equals}")
        self.outliers = pd.concat([self.outliers, df_outlier]).drop_duplicates()

    def showOutliers(self):
        if len(self.outliers) > 0: 
            return self.outliers

    def displayOutlierBounds(self):
        if len(self.outlierBounds) > 0:
            for item in self.outlierBounds:
                print(item)
        else:
            print('You have not specified any outlier bounds.')

    def targetBalance(self):
        return self.data[self.target].value_counts()

    def checkNulls(self):
        nullCount = []
        for column in self.data.columns:
            nullCount.append((column, self.data[column].isnull().sum()))
        return pd.DataFrame(nullCount, columns=["Column", "NullCount"]).sort_values("NullCount", ascending=False)

    def fullEDA(self, pairplot=False):
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()
        out5 = widgets.Output()
        out6 = widgets.Output()
        out7 = widgets.Output()
        out8 = widgets.Output()
        out9 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2, out3, out4, out5, out6, out7, out8, out9])
        tab.set_title(0, 'Info')
        tab.set_title(1, 'Categorical')
        tab.set_title(2, 'Numerical')
        tab.set_title(3, 'Pairplot')
        tab.set_title(4, 'Descriptive Statistics')
        tab.set_title(5, 'Correlations')
        tab.set_title(6, 'Outliers')
        tab.set_title(7, 'Target Balance')
        tab.set_title(8, "Null Count")
        display(tab)
            

        with out1:
            self.info()

        with out2:
            fig2 = self.countPlots(splitTarg=False, show=False)
            plt.show(fig2)
        
        with out3:
            fig3 = self.histPlots(kde=True, show=False)
            plt.show(fig3)

        with out4:
            if pairplot:
                fig4 = self.pairPlot(splitTarg=False)
                plt.show()

        with out5:
            display(self.describe())

        with out6:
            self.correlationMatrix()

        with out7:
            print('Outlier Conditions:')
            self.displayOutlierBounds()
            print()
            print('Outliers:')
            display(self.showOutliers())
            
        with out8:
            print("Target Balance:")
            display(self.targetBalance())

        with out9:
            display(self.checkNulls())


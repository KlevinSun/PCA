import numpy as np
import random
import matplotlib.pyplot as plt


def meanX(dataX):                               #calculate the mean of every column
    return np.mean(dataX, axis=0)


def pca(dataMat, k):                            #PCA process
    #print(dataMat)
    average = meanX(dataMat)
    m, n = np.shape(dataMat)
    data_adjust = []
    avgs = np.tile(average, (m, 1))

    data_adjust = dataMat - avgs                #Centralization

    covMat = np.cov(data_adjust.T)              #Calculate the covariance matrix

    eigVal, eigVects = np.linalg.eig(covMat)    #Calculate eigenvalues and eigenvectors
    eigValIndex = np.argsort(-eigVal)
    print("eigenvalues from max to min:")       #Sort eigenvalues and eigenvectors
    print(eigVal[eigValIndex[:13]])
    eigenValue = eigVal[eigValIndex[:13]]
    plot_eigenValue(eigenValue)

    finalData = []
    if k > n:
        print("k must lower than feature numbers")
        return
    else:
        selectVec = np.matrix(eigVects.T[eigValIndex[:k]])  #Get the transform matrix A
        print("The Matrix A:")
        print(selectVec.T)
        finalData = data_adjust*selectVec.T                 #Simplified k-dimensional data
        #print(finalData)
        reconData = (finalData * selectVec) +average        #Retrieve n-dimensional data
    return finalData, reconData


def compare(origin, recon):                         # Compare the origin data and retrieve data, get the average L2 distance
    m, n = np.shape(origin)
    p, q = np.shape(recon)
    if m != p or n != q:
        print("unable to compare")
        return
    result = 0
    for i in range(m):
        temp = 0
        for j in range(n):
            temp += (origin[i, j] - recon[i, j])**2
        result += np.sqrt(temp)
    result /= m
    return result

def plot_eigenValue(data):
    dataArr = np.array(data)

    m = np.shape(dataArr)[0]
    axis_x = []
    axis_y = []
    j = 1
    for i in range(m):
        axis_x.append(j)
        axis_y.append(np.log(dataArr[i]))
        j += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(axis_x, axis_y, s=50, c='red')
    plt.xlabel('from max to min'); plt.ylabel('log of eigenValue');
    plt.savefig("eigenValuePicture")
    plt.show()

class cluster:                                              #cluster process
    def __init__(self, data, k):
        m, n = np.shape(data)
        Data = [[0 for x in range(n)]for x in range(m)]
        for i in range(m):
            for j in range(n):
                Data[i][j] = data[i, j]
        #print(Data)
        self.data = Data
        self.k = k


    def init_center(self):
        m, n = np.shape(self.data)
        list = []
        for i in range(self.k):
            temp = random.randint(0, m-1)
            list.append(self.data[temp])
        return list


    def cal_distance(self, p1, p2):
        result = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
        return result

    def assort_node(self, center_list):
        templist = []
        for i in range(np.shape(self.data)[0]):
            mindis = None
            tempcenter = None
            for j in center_list:
                tempdis = self.cal_distance(self.data[i], j)
                if mindis == None:
                    mindis = tempdis
                    tempcenter = j
                elif mindis > tempdis:
                    mindis = tempdis
                    tempcenter = j
            templist.append((tempcenter, self.data[i]))
        return templist

    def cal_center(self, center_list, assort_result):
        temp_center = []
        for i in center_list:
            x = 0
            y = 0
            count = 0
            for j in assort_result:
                if str(i) == str(j[0]):
                    x += j[1][0]
                    y += j[1][1]
                    count += 1
            x /= count
            y /= count
            temp_center.append((x, y))
        return temp_center

    def compare_center(self, list1, list2):
        bool = False
        for i in list1:
            bool = False
            for j in list2:
                if str(i) == str(j):
                    bool = True
                    break
            if bool == False:
                break
        return bool

    def kmeansObject(self, assort_result):
        temp = 0
        count = 0
        for j in assort_result:
            t = self.cal_distance(j[0], j[1])
            temp += t**2
            count += 1
        return temp

    def kmeans(self):
        center_list = self.init_center()
        kmeansobject = 0
        while True:
            assort_result = self.assort_node(center_list)
            kmeansobject = self.kmeansObject(assort_result)
            this_center_list = self.cal_center(center_list, assort_result)
            if self.compare_center(this_center_list, center_list):
                return center_list, kmeansobject
            else:
                center_list = this_center_list



def main():
    dataMat = np.loadtxt(open("C:\\test\\wine.csv","rb"), delimiter=",", skiprows=0)

    k = 2
    finalData, reconMat = pca(dataMat, k)
    return finalData, reconMat, dataMat

if __name__ == "__main__":
    finalData,  reconMat, originData = main()
    average = compare(originData, reconMat)
    finalcenterlist = cluster.kmeans(cluster(finalData, k=3))
    print("centroids list:")
    print(finalcenterlist)
    print("L2 distance between origin data and retrieve data:")
    print(average)



import math
class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting
        self.maximumArticles = self.maxArticles()
        self.idfScore = self.computeIDF()


    # Method performing retrieval for specified query
    def forQuery(self, query):
        #Declare variables to be used in the program, adding 1 so that the index matches the articleID
        bestArticles, articleMatrix, frequencySum = [], ([0] * (self.maximumArticles + 1)), ([0] * (self.maximumArticles + 1))

        if (self.termWeighting == 'binary'):
            #Compute binary weight of each term for each article
            for key in query.keys():
                if key in self.index:
                    for article,count in self.index[key].items(): articleMatrix[article] += 1

            #Compute vector size
            for word in self.index.keys():
                for article, frequency in self.index[word].items():
                    frequencySum[article] += 1
            sqrFrequencySum = [math.sqrt(x) for x in frequencySum]

            # Compute Cosine similarity using Binary term weighting
            for number in range(1, self.maximumArticles): articleMatrix[number] = articleMatrix[number] / sqrFrequencySum[number]
            # return top articles
            return self.topArticles(articleMatrix)


        elif (self.termWeighting == 'tf'):
            #Computing the terms that are both in the query and article
            for word in query.keys():
                if word in self.index:
                    for article,frequency in self.index[word].items():
                        articleMatrix[article] += frequency*query[word]

            # Compute vector size
            for word in self.index.keys():
                for article, frequency in self.index[word].items():
                    frequencySum[article] += math.pow(frequency, 2)
            sqrFrequencySum =[math.sqrt(x) for x in frequencySum]

            # Compute Cosine similarity using TF term weighting
            for number in range(1, self.maximumArticles): articleMatrix[number] = articleMatrix[number] / sqrFrequencySum[number]
            # return top articles
            return self.topArticles(articleMatrix)

        elif (self.termWeighting == 'tfidf'):
            # Computing the terms that are both in the query and article
            for key in query.keys():
                if key in self.index:
                    for article, frequency in self.index[key].items():
                            articleMatrix[article] += frequency * query[key] * math.pow(self.idfScore[key],2)

            # Compute vector size
            for word in self.index.keys():
                for article, frequency in self.index[word].items():
                    frequencySum[article] += math.pow(frequency*(self.idfScore[word]), 2)
            sqrFrequencySum = [math.sqrt(x) for x in frequencySum]

            # Compute Cosine similarity using TFIDF term weighting
            for number in range(1, self.maximumArticles): articleMatrix[number] = articleMatrix[number] / sqrFrequencySum[number]
            # return top articles
            return self.topArticles(articleMatrix)

    #Extract the indexes of the top 10 scoring articles
    def topArticles(self,articleMatrix):
        bestArticles = []
        for x in range(10):
            indexOfMaxValue = articleMatrix.index(max(articleMatrix))
            bestArticles.append(articleMatrix.index(max(articleMatrix)))
            articleMatrix[indexOfMaxValue] = 0
        return bestArticles

    #Comnpute the maximum number of articles in the set
    def maxArticles(self):
        maximumArticles = 0
        for word in self.index.keys():
            for article in self.index[word]:
                if article > maximumArticles:
                    maximumArticles = article
        return maximumArticles

    #Compute the IDF weight
    def computeIDF(self):
        idfScore = {}
        for term in self.index:
            idfScore[term] = math.log10(self.maximumArticles/len(self.index[term].items()))
        return idfScore

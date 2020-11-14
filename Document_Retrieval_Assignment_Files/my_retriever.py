import math

class Retrieve:
    # Create new Retrieve object storing index and termWeighting scheme
    def __init__(self,index, termWeighting):
        self.index = index
        self.termWeighting = termWeighting
        self.docid_max = max([max(self.index[i]) for i in self.index])
        self.rangeNum = self.docid_max + 1
        self.allterms = {doc: {terms for terms, docs in index.items()
                                   if doc in docs}
                             for doc in range(1, self.rangeNum)}
        self.idf = self.IDF()
        
#=====================================================================================================================   
# helper 
    #Compute the IDF weight 
    def IDF(self):
        idf = {}
        for term in self.index:
            appearTime = len(self.index[term].items())
            idf[term] = math.log(len(self.allterms)/appearTime)
        return idf
    # Compute the sqrt of sum of frequency
    def frequencySum(self,docFrequency):
        Sum = []
        for x in docFrequency:
             tsum = math.sqrt(x)
             Sum.append(tsum)
        return Sum
   
    # Get top 10 scoring docs
    def tenDocs(self,docCount):
        tops = []
        for x in range(10):
            maxIndex = docCount.index(max(docCount))
            tops.append(maxIndex)
            docCount[maxIndex] = 0
        return tops         
#=====================================================================================================================
    # Method performing retrieval for specified query
    def forQuery(self, query):
        self.docCount= ([0] * self.rangeNum)
        docFrequency =([0] * self.rangeNum)
    #Calculate binary weight of the word in each document
        if self.termWeighting == 'binary':
            for word in query.keys():
                if word in self.index:
                    for doc, num in self.index[word].items():
                         self.docCount[doc] += 1
        # Returns a dictionary of size of vector
            for value in self.index.keys():
                for doc, ferq in self.index[value].items():
                    docFrequency[doc] += 1
 
            sqrSum = self.frequencySum(docFrequency)      
            # Compute similarity using Binary term weighting
            for x in range(1, self.rangeNum): 
                self.docCount[x] =  self.docCount[x] / sqrSum[x]
            return self.tenDocs(self.docCount)#return top 10 doc

#=====================================================================================================================
        elif self.termWeighting == 'tf':
            #Count words that exist in both the query and the document
            for word in query.keys():
                if word in self.index:
                    for doc,ferq in self.index[word].items():
                        self.docCount[doc] += ferq*query[word]
            # Returns a dictionary of size of vector
            for value in self.index.keys():
                for doc, ferq in self.index[value].items():
                    docFrequency[doc] += ferq **2 
                    
            sqrSum = self.frequencySum(docFrequency)  

            # Compute Cosine similarity using TF term weighting
            for x in range(1,self.rangeNum): 
                self.docCount[x] = self.docCount[x] / sqrSum[x]
                
            return self.tenDocs(self.docCount)
        
#=====================================================================================================================
        elif self.termWeighting == 'tfidf':
           #Count words that exist in both the query and the document
            for word in query.keys():
                if word in self.index:
                    for doc,ferq in self.index[word].items():
                            self.docCount[doc] += ferq * query[word] * (self.idf[word]**2)

           # Returns a dictionary of size of vector
            for value in self.index.keys():
                for doc, ferq in self.index[value].items():
                    docFrequency[doc] += math.pow(ferq*(self.idf[value]), 2)
                    
            sqrSum = self.frequencySum(docFrequency)  
            # Compute Cosine similarity using TFIDF term weighting
            for x in range(1, self.docid_max): 
                self.docCount[x] = self.docCount[x] / sqrSum[x]
 
            return self.tenDocs(self.docCount)


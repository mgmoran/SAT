import numpy as np
from collections import defaultdict
import scipy.linalg
from scipy import spatial
import random
import itertools
import re

## The DistributionalSemantics object takes an argument v as its parameter, corresponding to
## which word vector it will be using - Google or COMPOSES.

class DistributionalSemantics():
    def __init__(self,v):
        self.v = v
        self.wordindices = {}
        self.wordvectors = {}
        i = 0
        with open(v, 'r') as f:
            for line in f:
                line = line.split()
                word = line[0]
                word =  re.sub('-','',word)
                self.wordindices[word] = i
                self.wordvectors[i] = line
                i = i + 1

    def WordVector(self,filename):
        words = []
        wordcounts= defaultdict(int)
        indices = {}
        totalwords = 0
        # Grabbing word types and counts
        with open(filename, 'r') as f:
            for line in f:
                tokens = line.split()
                for token in tokens:
                    words.append(token)
                    wordcounts[token] += 1
                    totalwords += 1
        # Mapping tokens to indices
        i = 0
        for word in words:
            if word not in indices:
                indices[word] = i
                i+=1
        vocab = (len(set(words)))
        c = np.zeros((vocab, vocab))
        bigramdict = defaultdict(lambda: defaultdict(int))
        # Getting bigram counts for each word
        with open(filename, 'r') as f:
            for line in f:
                tokens = line.split()
                for i in range(len(tokens) - 1):
                        word = tokens[i]
                        index = indices[word]
                        if i < len(tokens) -1:
                            nextword = tokens[i + 1]
                            nindex = indices[nextword]
                            c[nindex][index] += 1
                            c[index][nindex] += 1
        # Multiplying c by 10 and appling add-1 smoothing
        c = c * 10
        c = c + 1
        ppmi = np.zeros((vocab,vocab))
        total = np.sum(c)
        # Filling in ppmi vector
        for w in indices:
            wprob = np.sum(c[indices[w]])/total
            for context in indices:
                cprob = np.sum(c[indices[context]])/total
                bigramprob = (c[indices[w],indices[context]]) / total
                ppmi[indices[w],indices[context]] = max((np.log2(bigramprob/(wprob * cprob))),0)
        print('vector distances')
        print('women and men: ' + repr(scipy.linalg.norm(ppmi[indices['women']] - ppmi[indices['men']])))
        print('women and dogs: ' + repr(scipy.linalg.norm(ppmi[indices['women']] - ppmi[indices['dogs']])))
        print('men and dogs: ' + repr(scipy.linalg.norm(ppmi[indices['men']] - ppmi[indices['dogs']])))
        print('feed and like: ' + repr(scipy.linalg.norm(ppmi[indices['feed']] - ppmi[indices['like']])))
        print('feed and bite: ' + repr(scipy.linalg.norm(ppmi[indices['feed']] - ppmi[indices['bite']])))
        print('like and bite: ' + repr(scipy.linalg.norm(ppmi[indices['like']] - ppmi[indices['bite']])))
        ### Singular Value Decomposition
        U, E, Vt = scipy.linalg.svd(ppmi,full_matrices=False)
        U = np.matrix(U)
        E = np.matrix(np.diag(E))
        Vt = np.matrix(Vt)
        V = Vt.T
        ppmi2 = np.dot(U,np.dot(E,Vt))
        # Checking that multiplying U, E and Vt obtains original ppmi matrix
        np.allclose(ppmi,ppmi2)
        # Reducing dimensionality of ppmi
        reduced_ppmi = ppmi * V[:, 0:3]
        print()
        print('reduced vector distances')
        print('women and men: ' + repr(scipy.linalg.norm(reduced_ppmi[indices['women']] - reduced_ppmi[indices['men']])))
        print('women and dogs: ' + repr(scipy.linalg.norm(reduced_ppmi[indices['women']] - reduced_ppmi[indices['dogs']])))
        print('men and dogs: ' + repr(scipy.linalg.norm(reduced_ppmi[indices['men']] - reduced_ppmi[indices['dogs']])))
        print('feed and like: ' + repr(scipy.linalg.norm(reduced_ppmi[indices['feed']] - reduced_ppmi[indices['like']])))
        print('feed and bite: ' + repr(scipy.linalg.norm(reduced_ppmi[indices['feed']] - reduced_ppmi[indices['bite']])))
        print('like and bite: ' + repr(scipy.linalg.norm(reduced_ppmi[indices['like']] - reduced_ppmi[indices['bite']])))

    def Synonymtest(self,filename,vector=None):
        if vector==None:
            vector = self.v
    ### Euclidean Distance Test
        with open(filename,'r') as f:
            questions = f.read().split("\n\n")
            correct = 0
            for question in questions:
                question = question.split("\n")
                if len(question)>1:
                    target = question[1]
                    # Normalizing the verbs to appear as they would in the Google/COMPOSES vectors
                    normalized = re.sub('to_','',target)
                    distances = []
                    answer = question[-1].split()[1]
                    options = question[2].split()
                    if normalized in self.wordindices:
                        index = self.wordindices[normalized]
                        word_vector = scipy.array(self.wordvectors[index][1:])
                        word_vector = word_vector.astype(float)
                        for option in options:
                            # Normalizing the verbs to appear as they would in the Google/COMPOSES vectors
                            option = re.sub('to_','',option)
                            if option not in self.wordindices:
                                distances.append(np.infty)
                            else:
                                index2 = self.wordindices[option]
                                option_vector =  scipy.array(self.wordvectors[index2][1:])
                                option_vector = option_vector.astype(float)
                                distances.append(scipy.linalg.norm(word_vector - option_vector))
                        guess = distances.index(min(distances))
                    else:
                        distances = [np.infty, np.infty, np.infty, np.infty, np.infty]
                        guess = distances.index(random.choice(distances))
                    if options[guess]==answer:
                        correct +=1
        Euclidean = (correct/1000)
    ### Cosine Similarity Test
        with open(filename, 'r') as f:
            questions = f.read().split("\n\n")
            correct = 0
            for question in questions:
                question = question.split("\n")
                # Handling the existence of some empty "questions" due to the way I processed the text
                if len(question)>1:
                    target = question[1]
                    # Normalizing the verbs to appear as they would in the Google/COMPOSES vectors
                    normalized = re.sub('to_', '', target)
                    similarities = []
                    answer = question[-1].split()[1]
                    options = question[2].split()
                    if normalized in self.wordindices:
                        index = self.wordindices[normalized]
                        word_vector = scipy.array(self.wordvectors[index][1:])
                        word_vector = word_vector.astype(float)
                        for option in options:
                            # Normalizing the verbs to appear as they would in the Google/COMPOSES vectors
                            option = re.sub('to_', '', option)
                            if option not in self.wordindices:
                                similarities.append(-1)
                            else:
                                index2 = self.wordindices[option]
                                option_vector = scipy.array(self.wordvectors[index2][1:])
                                option_vector = option_vector.astype(float)
                                similarities.append(1 - (spatial.distance.cosine(word_vector,option_vector)))
                        guess = similarities.index(max(similarities))
                    else:
                        similarities = [-1,-1,-1,-1,-1]
                        guess = similarities.index(random.choice(similarities))
                    if options[guess] == answer:
                        correct += 1
        Cosine = (correct/1000)
        return (Euclidean, Cosine)

    def AnalogyTest(self,filename,vector=None):
        if vector==None:
            vector = self.v
        with open(filename,'r') as f:
            analogies = defaultdict(dict)
            questions = f.read().split('\n\n')
            questions = [question for question in questions if not question[0][0]=='#']
            i = 0
            total = 0
            correct = 0
            count = 0
            for question in questions:
                question= question.split('\n')
                question = question[1:]
                target = (question[0].split())
                word1 = target[0]
                word2 = target[1]
                analogy = (word1,word2)
                if word1 in self.wordindices and word2 in self.wordindices:
                    index1 = self.wordindices[word1]
                    index2 = self.wordindices[word2]
                    word1_vector = scipy.array(self.wordvectors[index1][1:])
                    word1_vector = word1_vector.astype(float)
                    word2_vector = scipy.array(self.wordvectors[index2][1:])
                    word2_vector = word2_vector.astype(float)
                    a = question[1].split()[:-1]
                    b = question[2].split()[:-1]
                    c = question[3].split()[:-1]
                    d = question[4].split()[:-1]
                    e = question[5].split()[:-1]
                    answer = question[6]
                    analogies[analogy]['a'] = (a[0],a[1])
                    analogies[analogy]['b'] = (b[0], b[1])
                    analogies[analogy]['c'] = (c[0], c[1])
                    analogies[analogy]['d'] = (d[0], d[1])
                    analogies[analogy]['e'] = (e[0], e[1])
                    similarities = {}
                    for letter in analogies[analogy]:
                        choice = analogies[analogy][letter]
                        answer1 = choice[0]
                        answer2 = choice[1]
                        if answer1 not in self.wordindices or answer2 not in self.wordindices:
                            similarities[letter] = -1
                        else:
                            answerindex1 = self.wordindices[answer1]
                            answerindex2 = self.wordindices[answer2]
                            answer1_vector = scipy.array(self.wordvectors[answerindex1][1:])
                            answer1_vector = answer1_vector.astype(float)
                            answer2_vector = scipy.array(self.wordvectors[answerindex2][1:])
                            answer2_vector = answer2_vector.astype(float)
                            # Jurafsky method of obtaining a relation vector
                            subtraction = word1_vector - word2_vector
                            relation = subtraction + answer2_vector
                            similarities[letter] = (1 -spatial.distance.cosine(relation,answer1_vector))
                    similarities = sorted(similarities.items(), key=lambda item: (item[1]),reverse=True)
                    guess = similarities[0]
                else:
                    count +=1
                    guess = random.choice(['a','b','c','d','e'])
                if guess[0] == answer:
                    correct += 1
                total += 1
            return (correct/total)


if __name__=='__main__':
    d = DistributionalSemantics('EN-wform.w.2.ppmi.svd.500.rcv_vocab.txt')
    d.WordVector('dist_sim_data.txt')
    Euclidean,Cosine = d.Synonymtest('SynonymTestSet.txt')
    print("\nCOMPOSES Test Results")
    print("Synonym Test")
    print("Euclidean Distance accuracy: %.2f" % (Euclidean))
    print("Cosine Similarity accuracy: %.2f" % (Cosine))
    Analogy = d.AnalogyTest('SAT-package-V3.txt')
    print('Analogy Test: %.2f' % (Analogy))
    d = DistributionalSemantics('GoogleNews-vectors-rcv_vocab.txt')
    Euclidean,Cosine = d.Synonymtest('SynonymTestSet.txt')
    print("\nGoogle Test Results")
    print("Synonym Test")
    print("Euclidean Distance accuracy: %.2f" % (Euclidean))
    print("Cosine Similarity accuracy: %.2f" % (Cosine))
    Analogy = d.AnalogyTest('SAT-package-V3.txt')
    print('Analogy Test Results: %.2f' % (Analogy))

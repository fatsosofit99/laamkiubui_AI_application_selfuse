import jieba
from typing import List, Set


def load_stopwords(paths: List[str]) -> Set[str]:
    #TODO
    stopwords=set()
    for path in paths:
        with open(path,'r') as f:
            for line in f:
                word = line.strip()
                if word:
                    stopwords.add(word)
    return stopwords

def tokenize(text: str) -> List[str]:
    #TODO
    results=[]
    
    for word in jieba.lcut(text):
        if word.strip():
            results.append(word)
    return results


def filter_words(words: List[str], stopwords: Set[str]) -> List[str]:
    #TODO
    results=[]
    for word in words:
        if word not in stopwords:
            results.append(word)

    return results

if __name__ == "__main__":
    text = "我和我的祖国，是我永远的骄傲。"

    stopwords_files = ['/home/project/stopwords1.txt', '/home/project/stopwords2.txt']
    stopwords = load_stopwords(stopwords_files)
    words = tokenize(text)
    filtered = filter_words(words, stopwords)

    print(filtered)
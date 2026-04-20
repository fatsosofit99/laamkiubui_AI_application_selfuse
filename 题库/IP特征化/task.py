from gensim.models import KeyedVectors
from typing import List, Optional
import numpy as np
import pandas as pd

def ip_to_int(ip: str) -> int:
    parts=ip.split('.')
    return(int(parts[0])<<24)+(int(parts[1])<<16)+(int(parts[2])<<8)+int(parts[3])

    #TODO

def find_location_pandas(ip: str, csv_file: str) -> Optional[List[str]]:
    #TODO
    ip_num=ip_to_int(ip)
    df= pd.read_csv(csv_file,header=None,low_memory=False)
    df[0] = pd.to_numeric(df[0], errors='coerce')
    df[1] = pd.to_numeric(df[1], errors='coerce')
    match = df[(df[0]<=ip_num)&(df[1]>=ip_num)]
    if not match.empty:
        row = match.iloc[0]
        contry=str(row[3])
        region=str(row[4])
        city=str(row[5])
        def lower_first(s):
            return s[0].lower()+s[1:] if s else s
        return [lower_first(contry),lower_first(region),lower_first(city)]
    else:
        return None
def location_to_vector(location: List[str], file_path: str) -> np.ndarray:
    #TODO
    model = KeyedVectors.load_word2vec_format(file_path,binary=False,no_header=True)
    vectors=[]
    for word in location:
        word_lower = word.lower()
        if word_lower in model:
            vectors.append(model[word_lower])
        elif word in model:
            vectors.append(model[word])
    if vectors:
        return np.mean(vectors,axis=0)
    else:
        return np.zeros(model.vector_size)
def main() -> None:
    csv_file = 'IP2LOCATION.CSV'
    glove_file = 'glove.6B.50d.txt'
    ip_to_lookup = '1.0.0.1'

    location = find_location_pandas(ip_to_lookup, csv_file)

    if location:
        print(f"IP 地址 {ip_to_lookup} 的地理位置是：{location}")

        # 3. 使用 Gensim 将地理位置信息转换为向量
        location_vector = location_to_vector(location, glove_file)
        print(f"地理位置信息 '{location}' 的向量表示是：{location_vector}")
    else:
        print(f"未找到 IP 地址 {ip_to_lookup} 的地理位置信息")

if __name__ == '__main__':
    main()
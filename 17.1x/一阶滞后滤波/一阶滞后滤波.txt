from typing import List, Dict

def first_order_lag_filter(data: List[List[float]], alpha: float) -> List[List[float]]:
    res=[data[0][:]]
    for i in range(1,len(data)):
        res.append([alpha*data[i][j]+(1-alpha)*res[i-1][j]for j in range(2)])
    return res

    # TODO


def compare_statistics(original: List[List[float]], filtered: List[List[float]]) -> Dict[str, Dict[str, float]]:
    result={}
    def get_mu(col):
        return sum(col)/len(col)
    def get_dm(col):
        mu = get_mu(col)
        return sum((x - mu)**2 for x in col) / len(col)
    def get_R(col):
        return max(col)-min(col)
    for data, name in enumerate(['Pitch','Roll']):
        mo,vo,ro = get_mu([r[data] for r in original]),get_dm([r[data] for r in original]), get_R([r[data] for r in original])
        mf,vf,rf =get_mu([r[data] for r in filtered]),get_dm([r[data] for r in filtered]), get_R([r[data] for r in filtered])
        result[name]={'mean_diff':mo-mf,'var_diff':vo-vf,'range_diff':ro-rf}
    return result

    # TODO


if __name__ == '__main__':

    data = [
        [10.0, 5.0],
        [12.0, 8.0],
        [20.0, 7.0],
        [18.0, 10.0]
    ]
    alpha = 0.5

    filtered = first_order_lag_filter(data, alpha)
    print(filtered)
    result = compare_statistics(data, filtered)
    print(result)
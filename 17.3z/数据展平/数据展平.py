import os
import json
import csv
from typing import Dict, List, Any


def read_all_json_files_as_dict(folder_path: str, key_field: str) -> Dict[str, Dict[str, Any]]:
    #TODO
    result={}
    for filename in os.listdir(folder_path):
        file_path =os.path.join(folder_path,filename)
        
        with open(file_path,'r') as f:
            data = json.load(f)

        if key_field in data:
            result[data[key_field]]=data
    return result

        

def flatten_json(nested_dict: Dict[str, Any], sep: str = '.') -> Dict[str, Any]:
    #TODO
    """
    递归展开嵌套字典。例如：{'a': {'b': 1}} -> {'a.b': 1}
    """
    flat_dict = {}

    def _flatten(current_data: Any, prefix: str = ""):
        if isinstance(current_data, dict):
            for k, v in current_data.items():
                new_key = f"{prefix}{sep}{k}" if prefix else k
                _flatten(v, new_key)
        # 如果是列表，这里选择简单处理将其转为字符串，或根据需要进一步展开
        elif isinstance(current_data, list):
            flat_dict[prefix] = str(current_data)
        else:
            flat_dict[prefix] = current_data

    _flatten(nested_dict)
    return flat_dict

def merge_order_related_data(
    order_map: Dict[str, Dict],
    user_map: Dict[str, Dict],
    product_map: Dict[str, Dict],
    payment_map: Dict[str, Dict],
    logistics_map: Dict[str, Dict]
) -> List[Dict[str, Any]]:

    merged_rows = []

    for order_id, order in order_map.items():
        merged = {"订单号": order_id}
        merged.update(order)

        if order_id in payment_map:
            merged["支付信息"] = payment_map[order_id]

        if order_id in logistics_map:
            merged["物流信息"] = logistics_map[order_id]

        user_id = order.get("用户ID")
        if user_id and user_id in user_map:
            merged["用户信息"] = user_map[user_id]

        商品 = order.get("商品", {})
        商品ID = 商品.get("商品ID")
        if 商品ID and 商品ID in product_map:
            merged["商品信息"] = product_map[商品ID]

        merged_rows.append(merged)

    return merged_rows


def write_to_csv(dicts: List[Dict[str, Any]], output_file: str):
    #TODO
    all_keys=set()
    for d in dicts:
        all_keys.update(d.keys())
    header =sorted(list(all_keys))
    with open(output_file,'w')as f:
        writer=csv.DictWriter(f,fieldnames=header)
        writer.writeheader()
        writer.writerows(dicts)

if __name__ == '__main__':
    base_dir = "/home/project/json_data"

    order_map = read_all_json_files_as_dict(os.path.join(base_dir, "orders"), "订单号")
    user_map = read_all_json_files_as_dict(os.path.join(base_dir, "users"), "用户ID")
    product_map = read_all_json_files_as_dict(os.path.join(base_dir, "products"), "商品ID")
    payment_map = read_all_json_files_as_dict(os.path.join(base_dir, "payments"), "订单号")
    logistics_map = read_all_json_files_as_dict(os.path.join(base_dir, "logistics"), "订单号")

    merged_nested = merge_order_related_data(order_map, user_map, product_map, payment_map, logistics_map)

    flattened_rows = [flatten_json(row) for row in merged_nested]

    write_to_csv(flattened_rows, "flattened.csv")
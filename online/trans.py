import json

json_array = []
with open('dataset/online-uf/round2.jsonl','r') as file:
    for line in file:
        json_array.append(json.loads(line))

with open('dataset/online-uf/round2.json','w') as file:
    json.dump(json_array, file, indent=2)
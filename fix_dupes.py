import json

with open('agent_tasks.json', 'r') as f:
    data = json.load(f)

seen = set()
unique_tasks = []

for t in data['tasks']:
    if t['id'] not in seen:
        seen.add(t['id'])
        unique_tasks.append(t)

data['tasks'] = unique_tasks

with open('agent_tasks.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write('\n')

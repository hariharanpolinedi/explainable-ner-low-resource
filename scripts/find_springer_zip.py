import re

path = r'C:\Users\harih\.gemini\antigravity-ide\brain\04a763a0-d3e3-4528-ab2f-4852d12b7d2b\.system_generated\steps\221\content.md'
text = open(path, encoding='utf-8').read()

print('Searching for links...')
for m in re.finditer(r'href=[\'"]([^\'"]+)[\'"]', text):
    u = m.group(1)
    if any(k in u.lower() for k in ['zip', 'latex', 'author', 'template', 'springer']):
        print('Found link:', u)

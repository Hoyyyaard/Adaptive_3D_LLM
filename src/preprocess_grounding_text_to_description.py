import openai
import os
import json
from tqdm import tqdm
import random
import numpy


with open('/home/admin/Projects/EmbodiedScan/data/small_size_object/val_small_than_1e-3.json',"r") as f:
    data = json.load(f)

op_list = []
for dat in tqdm(data):
    if not 'point_within_bbox' in dat.keys() :
        continue
    descriptions = dat['text']
    obj_name = dat['target']
    des_prompt = ''
    descriptions = numpy.random.choice(descriptions, size=6)
    for i,t in enumerate(descriptions):
        des_prompt += f'{i+1}. {t}\n'

    prompt = f"To give you some examples of describing 3D interior scenes: \n\
Example 1: There is a dark brown wooden and leather chair. On the kitchen table.\n\
Example 2: The chair with the edge of the table facing the oven. lt will be the chainearthe railing.\n\
Example 3: This is a brown chair. It's on the left of the other chair.\n\
Now I have some description of an instance in an indoor scene, the description is used to locate the instance, please combine the description given to you, deduce the description of the instance but do not imagine, need to follow the style of the example\n\
Here is a description of the positioning of the {obj_name}:\n\
{des_prompt}\
Now please inference 2-4 detailed descriptions of this {obj_name}. Please do not use some grounding words like \"search\", \"look for\". Please do not use adversative relation in any description like \"but\", \"thus\". Please do not use negative statements. Just use the object name as subject. Each statement does not need to contain all the information in the description:"

    # print(prompt)

    # 调用GPT-4 API的函数
    def ask_GPT4(prompt): 
        result = openai.ChatCompletion.create(model="gpt-4",
                                    messages=[{"role": "user", "content": prompt}])
        print('answer:', result['choices'][0]['message']['content'])
        return result['choices'][0]['message']['content']

    # 调用函数
    ans = ask_GPT4(prompt)
    try:
        ans = ans.split('\n')
        ans = [a[3:] for a in ans]
        dat['answers'] = ans
        print(ans)
    except Exception as e:
        print(e)
    
    op_list.append(dat)


with open('/home/admin/Projects/EmbodiedScan/data/small_size_object/val_small_than_1e-3_wdes.json',"w") as f:
    json.dump(op_list, f)
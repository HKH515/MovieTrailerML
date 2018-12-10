import os
import json

frame_info = {
        'good': [],
        'zero': [],
        'long': [] # 10 mins is (framescount*100) / 30 > 600 circa
}

for i,movieid in enumerate(os.listdir("frames")):
    amount = len([n for n in os.listdir('frames/{}'.format(movieid)) if os.path.isfile('frames/{}/{}'.format(movieid, n))])
    if i % 100 == 0:
        print("processing {} fn {}".format(movieid, i))
    if amount == 0:
        frame_info['zero'].append(movieid)
    elif (amount*100)/30 > 200 or (amount*100)/30 < 40:
        frame_info['long'].append(movieid)
    else:
        frame_info['good'].append(movieid)

#frame_json = json.dumps(frame_info)

with open('frame_info.json', 'w') as f:
    json.dump(frame_info, f)

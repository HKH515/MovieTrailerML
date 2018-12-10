import os
import json

frame_info = {
        'good': [],
}

for i,movieid in enumerate(sorted(os.listdir("frames"))):
    amount = len([n for n in os.listdir('frames/{}'.format(movieid)) if os.path.isfile('frames/{}/{}'.format(movieid, n))])
    if i % 100 == 0:
        print("processing {} fn {}".format(movieid, i))
    if (amount*100)/30 > 200 or (amount*100)/30 < 40:
        pass
    else:
        frame_info['good'].append((movieid,amount))

#frame_json = json.dumps(frame_info)

with open('frame_info.json', 'w') as f:
    json.dump(frame_info, f)

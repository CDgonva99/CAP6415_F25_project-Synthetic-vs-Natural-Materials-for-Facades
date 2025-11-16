from pythonosc.udp_client import SimpleUDPClient
import json, time
c=SimpleUDPClient('127.0.0.1',8000)
classes=['brick','glass','concrete','metal','vegetation']
for i in range(20):
    msg=json.dumps({'label':classes[i%5],'confidence':0.8})
    c.send_message('/cv/pred', msg)
    print('sent', msg)
    time.sleep(0.3)

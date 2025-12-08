# === TouchDesigner reference callbacks (copy into a Text DAT in TD) ===

# 1) Button callback (bind this function to a Button COMP via DAT Execute or directly call from Panel Execute)
def save_and_send():
    # Save current TOP frame into runtime/td_input.png
    # Assumes there is a TOP named 'img_to_analyze' and a Movie File Out TOP named 'moviefileout1'
    # Make sure 'moviefileout1' path is set to: <repo>/runtime/td_input.png
    op('moviefileout1').par.record = True
    run("op('moviefileout1').par.record = False", delayFrames=2)  # stop after 2 frames

    # Send OSC /infer_path to Python at 127.0.0.1:8000
    from pythonosc import udp_client
    client = udp_client.SimpleUDPClient('127.0.0.1', 8000)
    repo = project.folder  # TD project folder
    # If moviefileout1 writes to absolute path, you can reuse that. Otherwise, construct relative path:
    img_path = repo + "/runtime/td_input.png"
    client.send_message("/infer_path", img_path)
    debug("Sent /infer_path", img_path)

# 2) OSC receive callback (for an OSC In DAT set to port 9000)
def onReceiveOSC(dat, rowIndex, message, bytes, timeStamp, address, args, peer):
    """
    Expect /cv with 7 numeric args:
      [0]=class_id, [1]=max_conf, [2:7]=brick..vegetation
    Writes into a Table DAT named 'cv_state' with rows:
      class_id, max_conf, brick, glass, concrete, metal, vegetation
    """
    if address != '/cv': return
    vals = [str(v) for v in args]
    t = op('cv_state')
    t.clear(); t.appendRow(['class_id','max_conf','brick','glass','concrete','metal','vegetation'])
    t.appendRow(vals)
    # Optional: drive a Switch TOP index using class_id
    try:
        cid = int(float(args[0]))
        op('facade_switch').par.index = cid
    except:
        pass


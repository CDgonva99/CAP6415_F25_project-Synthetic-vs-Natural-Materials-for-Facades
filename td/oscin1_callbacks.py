import json
def onReceiveOSC(dat, rowIndex, message, bytes, timeStamp, address, args, peer):
    try:
        payload = json.loads(args[0]) if args else {}
        t=op('cv_state')
        if not t: return
        t.clear(); t.appendRow(['label', payload.get('label','unknown')]); t.appendRow(['confidence', float(payload.get('confidence',0.0))])
    except Exception:
        pass
    return

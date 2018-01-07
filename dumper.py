import simplejson as json


def saveData(name, data):
    f = open(name, 'w')
    json.dump(data, f)
    f.close()


def loadData(name):
    with open(name) as f:
        data = json.load(f)
    return data
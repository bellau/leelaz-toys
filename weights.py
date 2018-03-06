import urllib2
import json
import shutil
import os.path
import gzip
import net_to_model as ntm

def model() :
    mf = load_model()
    return ntm.transform(mf)

def load_model():
    response = urllib2.urlopen("http://zero.sjeng.org/get-task/0")
    data = json.load(response)   
    net_hash = data["hash"]

    if( os.path.exists("./networks/%s" % net_hash)) :
        print "OK"
    else :
        # Open the url
        url = "http://zero.sjeng.org/networks/%s.gz" % ( net_hash )
        f = urllib2.urlopen( url)
        print "downloading " + url

        if not os.path.exists("./networks"):
            os.makedirs("./networks")
        # Open our local file for writing
        with open("./networks/%s.gz" % net_hash, "wb") as local_file:
            local_file.write(f.read())
        with gzip.open("./networks/%s.gz" % net_hash, "rb") as gzfile, open("./networks/%s" % net_hash, 'wb') as f:
            shutil.copyfileobj(gzfile, f)
    
    return "./networks/%s" % net_hash

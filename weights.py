import urllib2
import json
import shutil
import os.path
import gzip

def model():
    response = urllib2.urlopen("http://zero.sjeng.org/get-task/0")
    data = json.load(response)   
    net_hash = data["hash"]

    if( os.path.exists(net_hash)) :
        print "OK"
    else :
        # Open the url
        url = "http://zero.sjeng.org/networks/%s.gz" % ( net_hash )
        f = urllib2.urlopen( url)
        print "downloading " + url

        # Open our local file for writing
        with open("%s.gz" % net_hash, "wb") as local_file:
            local_file.write(f.read())
        with gzip.open("%s.gz" % net_hash, "rb") as gzfile, open(net_hash, 'wb') as f:
            shutil.copyfileobj(gzfile, f)
    
    return net_hash
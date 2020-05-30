from pydrive.drive import GoogleDrive 
from pydrive.auth import GoogleAuth 

# For using listdir() 
import os 


# Below code does the authentication 
# part of the code 
gauth = GoogleAuth() 

# Creates local webserver and auto 
# handles authentication. 
gauth.LocalWebserverAuth()	 
drive = GoogleDrive(gauth) 

# replace the value of this variable 
# with the absolute path of the directory 
path = r"C:\Users\703235761\Desktop\plans.txt"

f = drive.CreateFile({'title': 'plans.txt'}) 
f.SetContentFile( path) 
f.Upload() 

# Due to a known bug in pydrive if we 
# don't empty the variable used to 
# upload the files to Google Drive the 
# file stays open in memory and causes a 
# memory leak, therefore preventing its 
# deletion 
f = None

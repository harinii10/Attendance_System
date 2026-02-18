import urllib.request

url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
filename = "haarcascade_frontalface_default.xml"

print(f"Downloading {filename}...")
urllib.request.urlretrieve(url, filename)
print("Download complete.")

import bs4 as bs
import requests


def save_all(url, i, num):

	try:
		r = requests.get(url)
		result = r.text
		soup = bs.BeautifulSoup(result, 'html.parser')
		txt = soup.encode('ascii')

		if num == 0:
			fw = open("./saved_files/saved"+str(i)+".html", "wb")
		else:
			fw = open("./saved_files/saved"+str(i)+"_" + str(num) + ".html", "wb")
		fw.write(txt)
		fw.close()
		return 1

	except Exception as e:
		print(e)
		return 0

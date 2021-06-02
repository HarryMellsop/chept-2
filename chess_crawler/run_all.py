import pickle
import tqdm
import time
from save_rendered_webpage import save_all
from multiprocessing import Pool

def download(input):
	i, link, extra_links = input
	total = 0
	success = 0

	# if i < resume_from: return 0
	time.sleep(2)

	num = extra_links[i]
	for j in range(num + 1): # j=0 is the home page. Begin range from 0 if home page is to be included.
		if j != 0:
			url = link + "&pg="+str(j)
		else:
			url = link

		result = save_all(url, i, j)

		# retry once if we fail 
		if result == 0:
			print('Retrying')
			time.sleep(5)
			result = save_all(url, i, j)
		success += result
		if success != 0 and i % 10 == 0:
			print("Weeee just saved another one boys")
		total += 1
	return (total, success)

def main():
	start = 0
	end = 11577
	resume_from = 0

	all_links = pickle.load(open("./saved_files/saved_links.p", "rb") )
	extra_links = pickle.load(open("extra_pages.p", "rb") )
	print('Number of pages: ',  len(all_links))

	tot = 0
	succ = 0

	input = [(i, link, extra_links) for i, link in enumerate(all_links)]

	# Uncomment if your name is HARRY'S DESKTOP
	# input = input[0: len(input) // 4]
	# Uncomment if your name is HARRY'S LAPTOP
	# input = input[len(input) // 4: len(input) // 2]
	# Uncomment if your name is COLLIN'S DESKTOP
	# input = input[len(input) // 2: 3 * len(input) // 4]
	# Uncomment if your name is COLLIN'S LAPTOP
	input = input[3 * len(input) // 4:]

	

	# Uncomment 
	print(f"Processing {len(input)} links...")

	out = list(map(download, tqdm.tqdm(input)))

	tot = sum([elem[0] for elem in out])
	succ = sum([elem[1] for elem in out])


	print('Error percent: ', ((tot - succ) / succ))

if __name__ == "__main__":
	main()
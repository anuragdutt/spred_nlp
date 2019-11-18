import os
import sys
import numpy as np
import pandas as pd
import re



class extract8k(object):
	def __init__(self, dt):
		self.dt = dt

	def filingExtractor(tickers):
		try:
			base_url = "https://www.sec.gov/cgi-bin/browse-edgar"
			inputted_cik = cik
			payload = {
				"action" : "getcompany",
				"CIK" : inputted_cik,
				"type" : "8-K",
				"output":"xml",
				"dateb" : self.dt,
			}
			sec_response = requests.get(url=base_url,params=payload)
			soup = BeautifulSoup(sec_response.text,'lxml')
			url_list = soup.findAll('filinghref')
			html_list = []
            # Get html version of links
			for link in url_list:
				link = link.string
				if link.split(".")[len(link.split("."))-1] == 'htm':
					txtlink = link + "l"
					html_list.append(txtlink)

			doc_list = []
			doc_name_list = []
           	# Get links for txt versions of files
			for k in range(len(html_list)):
				txt_doc = html_list[k].replace("-index.html",".txt")
				doc_name = txt_doc.split("/")[-1]
				doc_list.append(txt_doc)
				doc_name_list.append(doc_name)
                # Create dataframe of CIK, doc name, and txt link
			df = pd.DataFrame(
			{
				"cik" : [cik]*len(html_list),
				"ticker" : [ticker]*len(html_list),
				"txt_link" : doc_list,
				"doc_name": doc_name_list
			})
		except requests.exceptions.ConnectionError:
			sleep(.1)
		return df



if __name__ == "__main__":

	wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
	cik_df = pd.read_html(wiki_url,header=[0],index_col=0)[0]
	tickers = cik_df.index.drop_duplicates().values
	print(tickers)

	# Get table of the S&P 500 tickers, CIK, and industry from Wikipedia
	wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
	cik_df = pd.read_html(wiki_url,header=0,index_col=0)[0]
	cik_df['GICS Sector'] = cik_df['GICS Sector'].astype("category")
	cik_df['GICS Sub Industry'] = cik_df['GICS Sector'].astype("category")
	print(cik_df.shape)
	print(cik_df.head())



# 	fn = "../data/tmp/AAPL.gz"
# 	# with open(fn, 'rb') as f:
# 	# 	content = f.read()



# 	# for line in content:
# 	# 	print(line)

# 	# timestamp = [int(line[5:-1]) for line in gzip.open(fn, 'rb') if line.startswith('TIME:')][::-1]
# 	# [print(line) for line in open(fn)][::-1]

# 	f = gzip.open(fn, 'r')
# 	# file_content = f.read()
	
# 	[print(line) for line in file_content]
# 	# timestamp = [print(line) for line in file_content if line.startswith('TIME:')][::-1]


# 	print(timestamp[1])
# 	exit(0)
# 	f = getFeatures(fn)
# 	print(f)



# def getFeatures(file):
#     text = open(file).read()

#     events = re.compile('<DOCUMENT>(.*?)</DOCUMENT>', re.DOTALL).findall(text)
#     print(type(events))
#     print(events[0])
#     exit(0)


#     events = [' '.join(event.split()) for event in events]    

#     count_vect = CountVectorizer()
#     word_counts = count_vect.fit_transform(events)    

#     print(word_counts)
#     exit(1)

#     tfidf_transformer = TfidfTransformer()
#     features = tfidf_transformer.fit_transform(word_counts)

#     return features

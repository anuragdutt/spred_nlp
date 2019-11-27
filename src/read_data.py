import os
import sys
import numpy as np
import pandas as pd
import re
from bs4 import BeautifulSoup
import requests
import pickle
from time import sleep
import datetime
import unicodedata

class extract8k(object):
	def __init__(self, dt):
		self.dt = dt

	def filingExtractor(self, cik, ticker):
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


	def extractText(self, link):
		try:
			r = requests.get(link)
			#Parse 8-K document
			filing = BeautifulSoup(r.content,"html5lib",from_encoding="ascii")
			#Extract datetime
			try:
				submission_dt = filing.find("acceptance-datetime").string[:14]
			except AttributeError:
			        # Flag docs with missing data as May 1 2018 10AM
				submission_dt = "20190501100000"

			submission_dt = datetime.datetime.strptime(submission_dt,"%Y%m%d%H%M%S")
			#Extract HTML sections
			for section in filing.findAll("html"):
			    #Remove tables
				for table in section("table"):
					table.decompose()
			    #Convert to unicode
				section = unicodedata.normalize("NFKD",section.text)
				section = section.replace("\t"," ").replace("\n"," ").replace("/s"," ").replace("\'","'")
			filing = "".join((section))
		except requests.exceptions.ConnectionError:
			sleep(10)
		
		sleep(.1)

		return filing, submission_dt

	def extractItemNo(self, document):
		pattern = re.compile("Item+ +\d+[\:,\.]+\d+\d")
		item_list = re.findall(pattern,document)
		return item_list



if __name__ == "__main__":

	save_toggle = 1
	pfn = "../data/pickles/df_sec_links.pickle"
	dt = "20190501"
	
	ptf = "../data/pickles/df_sec_text.pickle"

	sec_ext = extract8k(dt)

	if save_toggle == 0:

		wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
		cik_df = pd.read_html(wiki_url,header=[0],index_col=0)[0]
		tickers = cik_df.index.drop_duplicates().values


		# Get table of the S&P 500 tickers, CIK, and industry from Wikipedia
		wiki_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
		cik_df = pd.read_html(wiki_url,header=0,index_col=0)[0]
		cik_df['GICS Sector'] = cik_df['GICS Sector'].astype("category")
		cik_df['GICS Sub Industry'] = cik_df['GICS Sector'].astype("category")
		
		cik_df['ticker'] = tickers
		cik_df['symbol'] = cik_df.index.values
		# cik_df = cik_df[:5]
		# print(cik_df)


		df_links = []

		for index,row in cik_df.iterrows():
			tick = row['ticker']
			print(tick)
			cik = row['CIK']
			links = sec_ext.filingExtractor(cik, tick)
			df_links.append(links)

		fw = open(pfn, 'wb')
		pickle.dump(df_links, fw)
		fw.close()

	with open(pfn, "rb") as fr:
		df_links = pickle.load(fr)

	df_text = []

	for df in df_links:
		print(df['ticker'].unique())
		df['text'], df['release_date'] = zip(*df['txt_link'].apply(sec_ext.extractText))
		df['items'] = df['text'].map(sec_ext.extractItemNo)
		df_text.append(df)
	
	ffw = open(ptf, 'wb')
	pickle.dump(df_text, ffw)
	ffw.close()

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

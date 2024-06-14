# ~ Import packages ~ #
from google_patent_scraper import scraper_class
import pandas as pd
import json

import csv
import simplejson as sjson

# ~ Initialize scraper class ~ #
scraper=scraper_class() 

# ~~ Scrape patents individually ~~ #
patent_1 = 'WO2022060836A1'

err_1, soup_1, url_1 = scraper.request_single_patent(patent_1)
a=scraper.process_patent_html(soup_1)
c=a['claims_total']
# print(f'{c}')
# ~ Parse results of scrape ~ #
patent_1_parsed = scraper.get_scraped_data(soup_1,patent_1,url_1)



with open('test.json', 'w') as fp:
    json.dump(a, fp)

pdObj = pd.read_json('test.json', orient='index')
pdObj.to_csv('streaming.csv')


# # 定义转换函数
# def trans(json_path, csv_path):
#     json_file = open(json_path, 'r', encoding='utf8') 
#     csv_file = open(csv_path, 'a', newline='', encoding='utf8')
#     writer = csv.writer(csv_file)
#     # 读取json数据
#     dic_data = sjson.load(json_file)

#     keys = []
#     for dic in dic_data:
#         keys = dic.keys()
#         # 写入列名
#         writer.writerow(keys)
#         break

#     for dic in dic_data:
#         for key in keys:
#             if key not in dic:
#                 dic[key] = ''
#         writer.writerow(dic.values())

#     json_file.close()
#     csv_file.close()

# # 调用转换函数
# json_path = 'test.json'
# csv_path = 'streaming.csv'
# trans(json_path, csv_path)



#patent_2 = 'WO2022060836A1'

#err_2, soup_2, url_2 = scraper.request_single_patent(patent_2)

# ~ Parse results of scrape ~ #
#patent_2_parsed = scraper.get_scraped_data(soup_2,patent_2,url_2)

#data = patent_2_parsed['forward_cite_no_family']

#import json
#output = json.loads(data)
#
#df = pd.DataFrame(output)

# Specify the desired CSV file path
#csv_file = f'{patent_2}_cite_by.csv'

# Save DataFrame to CSV
#df.to_csv(csv_file, index=False)

#print(f"Output saved as {csv_file}")

import pandas as pd
#!/usr/bin/env python

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.colors as colors
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import json
import socket, struct

def optimise_k_means(data,max_k):

	means = []
	inertias = []

	for k in range(1,max_k):
		kmeans = KMeans(n_clusters=k)
		kmeans.fit(data)
		means.append(k)
		inertias.append(kmeans.inertia_)

	fig = plt.subplots(figsize=(10,5))
	plt.plot(means, inertias, 'o-')
	plt.xlabel("Number of Clusters")
	plt.ylabel("Inertia")
	plt.grid(True)
	plt.show()


#with open('/home/gonzalo/Escritorio/IBERDROLA/dataset.json') as f:
#    data = json.load(f)

def cluster():
	data = pd.read_json('/home/gonzalo/Escritorio/IBERDROLA/dataset.json')

	columns = []
	for col in data.columns:
	    columns.append(col)

	data = data.to_numpy()

	for d in data:
		try:
			d[0]=int.from_bytes(bytes(d[0], 'utf-8'), byteorder='big')
		except:
			d[0]=d[0]	
		d[1]=int.from_bytes(bytes(d[1], 'utf-8'), byteorder='big')
		d[2]=0
		d[3]=int.from_bytes(bytes(d[3], 'utf-8'), byteorder='big')
		d[4]=int.from_bytes(bytes(d[4], 'utf-8'), byteorder='big')
		d[5]=int.from_bytes(bytes(d[5], 'utf-8'), byteorder='big')
		d[6]=0
		d[7]=int.from_bytes(bytes(d[7], 'utf-8'), byteorder='big')
		d[9]=int.from_bytes(bytes(d[9], 'utf-8'), byteorder='big')


	df = pd.DataFrame(data=data,columns=columns)

	#optimise_k_means(df[['alert','platform','destination-ip','source-ip','time','priority','hostname']],30)
	#optimise_k_means(df[['priority','time']],30)
	kmeans = KMeans(n_clusters=9)
	kmeans.fit(df[['priority','time']])
	#kmeans.fit(df[['alert','platform','destination-ip','source-ip','time','priority','hostname']])
	df['KMeans'] = kmeans.labels_

	#print(df)

	df1 = df[['platform','priority','time','KMeans']]

	df1.to_csv('/home/gonzalo/Escritorio/IBERDROLA/cluster.csv')

	
	labels = kmeans.predict(df[['priority','time']])
	C = kmeans.cluster_centers_
	colores=['red','green','blue','cyan','yellow','purple','brown','black','orange']
	asignar=[]
	for row in labels:
	    asignar.append(colores[row])


	# Getting the values and plotting it
	f1 = df1['time'].values
	f2 = df1['priority'].values
	 
	plt.scatter(f1, f2, c=asignar, s=70)
	plt.scatter(C[:, 0], C[:, 1], marker='*', c=colores, s=1000)
	plt.show()



	



if __name__=="__main__":
	cluster()
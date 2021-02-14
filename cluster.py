import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
import matplotlib.cm as cm

#Read in smiles file to pandas dataframe
table = pd.DataFrame()
table = pd.read_csv('./data/all_list.csv')

#Some SMILES are pains in the ass, so we drop whatever SMILES give RDKIT kekulization/valence errors, etc.
for i in table.index:
    try:
        mol=Chem.MolFromSmiles(table.loc[i,'SMILES'], True)
        table.loc[i, 'MolWt'] = Descriptors.ExactMolWt(mol)
        table.loc[i, 'TPSA'] = Chem.rdMolDescriptors.CalcTPSA(mol)  # Topological Polar Surface Area
        table.loc[i, 'nRotB'] = Descriptors.NumRotatableBonds(mol)  # Number of rotable bonds
        table.loc[i, 'HBD'] = Descriptors.NumHDonors(mol)  # Number of H bond donors
        table.loc[i, 'HBA'] = Descriptors.NumHAcceptors(mol)  # Number of H bond acceptors
        table.loc[i, 'LogP'] = Descriptors.MolLogP(mol)  # LogP
    except:
        table.drop([i])

table = table.dropna()
descriptors = table.loc[:, ['MolWt', 'TPSA', 'nRotB', 'HBD','HBA', 'LogP']].values
descriptors_std = StandardScaler().fit_transform(descriptors)
pca = PCA()
descriptors_2d = pca.fit_transform(descriptors_std)
descriptors_pca= pd.DataFrame(descriptors_2d)
descriptors_pca.index = table.index
descriptors_pca.columns = ['PC{}'.format(i+1) for i in descriptors_pca.columns]
descriptors_pca.head(10)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))
plt.rcParams['axes.linewidth'] = 1
plt.figure(figsize=(8,6))
fig, ax = plt.subplots(figsize=(8,6))

var=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=3)*100)
plt.plot([i+1 for i in range(len(var))],var,'k-',linewidth=2)
plt.xticks([i+1 for i in range(len(var))])
plt.ylabel('% Variance Explained',fontsize=16,fontweight='bold')
plt.xlabel('Principal Component (PC)',fontsize=16,fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.tick_params ('both',width=2,labelsize=12)
plt.savefig('./models/variance_ratio.png', dpi=300)
plt.show()

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

#Principal component analysis for PC1 and PC2
ax.plot(descriptors_pca['PC1'],descriptors_pca['PC2'],'o',color='k', markersize=3.25, alpha=0.7)
ax.set_title ('Principal Component Analysis [PC1 + PC2]',fontsize=16,fontweight='bold',family='sans-serif')
ax.set_xlabel ('PC1',fontsize=14,fontweight='bold')
ax.set_ylabel ('PC2',fontsize=14,fontweight='bold')
plt.tick_params ('both',width=2,labelsize=12)
plt.tight_layout()
plt.savefig('./models/pca.png', dpi=300)
plt.show()

# This normalization will be performed just for PC1 and PC2, but can be done for all the components.
scale1 = 1.0/(max(descriptors_pca['PC1']) - min(descriptors_pca['PC1']))
scale2 = 1.0/(max(descriptors_pca['PC2']) - min(descriptors_pca['PC2']))
# And we add the new values to our PCA table
descriptors_pca['PC1_normalized']=[i*scale1 for i in descriptors_pca['PC1']]
descriptors_pca['PC2_normalized']=[i*scale2 for i in descriptors_pca['PC2']]

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)

ax.plot(descriptors_pca['PC1_normalized'],descriptors_pca['PC2_normalized'],'o',color='k', markersize=3.25, alpha=0.7)
ax.set_title ('Principal Component Analysis, Normalized',fontsize=16,fontweight='bold',family='sans-serif')
ax.set_xlabel ('PC1',fontsize=14,fontweight='bold')
ax.set_ylabel ('PC2',fontsize=14,fontweight='bold')
plt.tick_params ('both',width=2,labelsize=12)
plt.tight_layout()
plt.savefig('./models/pca_normalized.png', dpi=300)
plt.show()

range_n_clusters = np.arange(2,11,1)
for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(8, 4)
    kmeans = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = kmeans.fit_predict(descriptors_pca[['PC1_normalized', 'PC2_normalized']])
    silhouette_avg = silhouette_score(descriptors_pca[['PC1_normalized', 'PC2_normalized']], cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    sample_silhouette_values = silhouette_samples(descriptors_pca[['PC1_normalized', 'PC2_normalized']], cluster_labels)

    y_lower = 10

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(descriptors_pca['PC1_normalized'], descriptors_pca['PC2_normalized'],
                marker='.', s=10, lw=0, alpha=0.5, c=colors, edgecolor='k')

    # Labeling the clusters
    centers = kmeans.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    plt.suptitle(("Silhouette analysis with n clusters = %d" % n_clusters),
                 fontsize=10, fontweight='bold')
    plt.savefig('./models/silhouette_%d' % n_clusters, dpi=300)

plt.show()

kmeans = KMeans(n_clusters=6) # We define the best number of clusters here
clusters = kmeans.fit(descriptors_pca[['PC1_normalized','PC2_normalized']]) #PC1 vs PC2 (normalized values)
descriptors_pca['Cluster_PC1_PC2'] = pd.Series(clusters.labels_, index=table.index)
plt.rcParams['axes.linewidth'] = 1.5
plt.figure(figsize=(10, 8))

fig, ax = plt.subplots(figsize=(7, 7))

color_code = {0: 'magenta', \
              1.0: 'orange', \
              2.0: 'cyan', \
              3.0: 'c', \
              4.0: 'm', \
              5.0: 'y', \
              6.0: 'darkorange',
              7.0: 'k',
              8.0: 'blue',
              9.0: 'red'
              }

for i in descriptors_pca.index:
    ax.plot(descriptors_pca.loc[i].at['PC1_normalized'], descriptors_pca.loc[i].at['PC2_normalized'],
            c=color_code[descriptors_pca.loc[i].at['Cluster_PC1_PC2']],
            marker='o', markersize=8, markeredgecolor='k', alpha=0.3)

plt.xlabel('PC1', fontsize=14, fontweight='bold')
ax.xaxis.set_label_coords(0.98, 0.45)
plt.ylabel('PC2', fontsize=14, fontweight='bold')
ax.yaxis.set_label_coords(0.45, 0.98)
plt.tick_params('both', width=2, labelsize=12)
ax.spines['left'].set_position(('data', 0))
ax.spines['bottom'].set_position(('data', 0))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

lab = ['MolWt', 'TPSA', 'nRotB', 'HBD', 'HBA', 'LogP']  # Feature labels

l = np.transpose(pca.components_[0:2, :])  ## We will get the components eigenvectors (main features) for PC1 and PC2

n = l.shape[0]
for i in range(n):
    plt.arrow(0, 0, l[i, 0], l[i, 1], color='k', alpha=0.6, linewidth=1.2, head_width=0.025)
    plt.text(l[i, 0] * 1.25, l[i, 1] * 1.25, lab[i], color='k', va='center', ha='center', fontsize=11)

circle = plt.Circle((0, 0), 1, color='gray', fill=False, clip_on=True, linewidth=1.5, linestyle='--')
ax.add_artist(circle)
plt.xlim(-0.8, 0.8)
plt.ylim(-0.8, 0.8)
plt.tight_layout()
plt.savefig('./models/plot_analysis.png', dpi=300)
plt.show()
table=table.join(descriptors_pca)
table.to_csv('./data/chem_data.csv')

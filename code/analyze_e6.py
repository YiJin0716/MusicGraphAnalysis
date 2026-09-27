"""Weighted Leiden E6 comparison on the saved k=3 and k=64 graphs.

Run: python code/analyze_e6.py
Reads saved graphs and E5 results in results/. Writes to results/E6/.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / 'results' / 'E6'

import igraph as ig
import leidenalg as la
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

RUN = HERE.parent
E4_E5 = RUN / 'E4_E5_extension'
SEEDS = tuple(range(521, 526))
KS = (3, 64)
from matplotlib.ticker import MaxNLocator
plt.rcParams.update({'font.family': 'serif',
                     'font.serif': ['DejaVu Serif'],
                     'font.size': 10, 'mathtext.fontset': 'cm', 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 140})


def write_csv(path, header, records):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(records)


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')


def load_graph(k):
    path = RUN / 'graphs' / f'k{k:03d}.npz'
    a = sparse.load_npz(path).tocsr()
    a.sort_indices()
    assert a.shape == (2000, 2000) and (a != a.T).nnz == 0
    assert np.all(a.diagonal() == 0) and np.all(np.isfinite(a.data)) and np.all(a.data > 0)
    upper = sparse.triu(a, k=1).tocoo()
    edges = list(zip(upper.row.tolist(), upper.col.tolist()))
    weights = upper.data.astype(float).tolist()
    graph = ig.Graph(n=a.shape[0], edges=edges, directed=False)
    graph.es['weight'] = weights
    assert graph.ecount() == upper.nnz and graph.is_connected()
    return a, upper, graph, hashlib.sha256(path.read_bytes()).hexdigest()


def canonicalize(membership):
    groups = {}
    for index, community in enumerate(membership):
        groups.setdefault(int(community), []).append(index)
    ordered = sorted(groups.values(), key=lambda g: (-len(g), min(g)))
    result = np.empty(len(membership), dtype=int)
    for new_id, group in enumerate(ordered, 1):
        result[group] = new_id
    return result


def genre_agreement(upper, assignment, tag_sets):
    both = np.array([bool(tag_sets[i]) and bool(tag_sets[j])
                     for i, j in zip(upper.row, upper.col)])
    shared = np.array([bool(tag_sets[i] & tag_sets[j])
                       for i, j in zip(upper.row, upper.col)])
    within = assignment[upper.row] == assignment[upper.col]
    result = {}
    for name, mask in (('within', within), ('between', ~within)):
        eligible = mask & both
        result[name + '_labeled_edges'] = int(eligible.sum())
        result[name + '_shared_genre_edges'] = int((eligible & shared).sum())
        result[name + '_shared_genre_fraction'] = (float((eligible & shared).sum() / eligible.sum())
                                                  if eligible.any() else None)
    return result


def community_statistics(a, upper, assignment, e5, nodes, tag_sets, k, folder):
    n_communities = int(assignment.max())
    total_strength = float(upper.data.sum() * 2)
    rows = []
    contingency = np.zeros((n_communities, 2), dtype=int)
    for c in range(1, n_communities + 1):
        idx = np.flatnonzero(assignment == c)
        mask = assignment[upper.row] == c
        internal = mask & (assignment[upper.col] == c)
        touching = mask | (assignment[upper.col] == c)
        boundary = touching & ~internal
        volume = float(a[idx].sum())
        internal_weight = float(upper.data[internal].sum())
        boundary_weight = float(upper.data[boundary].sum())
        assert abs(volume - (2 * internal_weight + boundary_weight)) < 1e-7
        cc = connected_components(a[idx][:,idx], directed=False)[0]
        assert cc == 1, f'Community {c} in k={k} is disconnected'
        artists = Counter(nodes[i]['artist_id'] for i in idx)
        contingency[c-1] = np.bincount(e5[idx] - 1, minlength=2)
        rows.append([c, len(idx), int(internal.sum()), int(boundary.sum()),
                     internal_weight, boundary_weight, volume,
                     boundary_weight / volume, len(artists), max(artists.values()),
                     max(artists.values()) / len(idx),
                     int(contingency[c-1,0]), int(contingency[c-1,1]),
                     int(idx.min())])
    write_csv(folder / 'communities.csv',
              ['community_id','nodes','internal_edges','boundary_edges','internal_weight',
               'boundary_weight','volume','boundary_over_volume','artist_count',
               'largest_artist_count','largest_artist_fraction','e5_domain_1',
               'e5_domain_2','min_global_node_index'],rows)
    write_csv(folder / 'e5_overlap.csv',
              ['community_id','e5_domain_1','e5_domain_2','share_in_domain_1',
               'share_in_domain_2'],
              ((i+1,int(v[0]),int(v[1]),v[0]/sum(v),v[1]/sum(v))
               for i,v in enumerate(contingency)))
    tags = sorted({tag for group in tag_sets for tag in group})
    labeled = np.array([bool(group) for group in tag_sets])
    tag_total = Counter(tag for group in tag_sets for tag in group)
    tag_rows = []
    for c in range(1,n_communities+1):
        idx = np.flatnonzero((assignment==c) & labeled)
        counts = Counter(tag for i in idx for tag in tag_sets[i])
        for tag in tags:
            overall = tag_total[tag] / int(labeled.sum())
            p = counts[tag] / len(idx) if len(idx) else None
            tag_rows.append([c,tag,len(idx),counts[tag],p,overall,
                             p/overall if p is not None else None])
    write_csv(folder / 'genre_by_community.csv',
              ['community_id','genre','genre_labeled_nodes','hits','fraction',
               'whole_labeled_sample_fraction','lift'],tag_rows)
    return rows, contingency, tags, tag_rows


def extra_interpretation(assignment, nodes, folder):
    names=json.loads((RUN/'feature_names.json').read_text(encoding='utf-8'))
    raw=np.load(RUN/'features_raw.npy',allow_pickle=False)
    onset=raw[:,names.index('rhythm.onset_rate')]
    bpm=raw[:,names.index('rhythm.bpm')]
    write_csv(folder/'community_features.csv',
              ['community_id','nodes','mean_onset_rate','median_onset_rate',
               'mean_bpm','median_bpm'],
              ((c,int((assignment==c).sum()),float(onset[assignment==c].mean()),
                float(np.median(onset[assignment==c])),float(bpm[assignment==c].mean()),
                float(np.median(bpm[assignment==c])))
               for c in range(1,int(assignment.max())+1)))
    labeled=np.array([bool(n['mood/theme']) for n in nodes])
    tag_sets=[set(n['mood/theme']) for n in nodes]
    tags=sorted({tag for group in tag_sets for tag in group})
    totals=Counter(tag for group in tag_sets for tag in group)
    table=[]
    for c in range(1,int(assignment.max())+1):
        idx=np.flatnonzero((assignment==c)&labeled)
        counts=Counter(tag for i in idx for tag in tag_sets[i])
        for tag in tags:
            base=totals[tag]/int(labeled.sum())
            value=counts[tag]/len(idx) if len(idx) else None
            table.append([c,tag,len(idx),counts[tag],value,base,
                          value/base if value is not None else None])
    write_csv(folder/'mood_theme_by_community.csv',
              ['community_id','mood_theme','labeled_nodes','hits','fraction',
               'whole_labeled_sample_fraction','lift'],table)


def save_plot(fig, folder, stem):
    fig.savefig(folder / f'{stem}.png', dpi=190, bbox_inches='tight')
    fig.savefig(folder / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_results(k, folder, rows, contingency, tag_rows, spectral):
    sizes = np.array([r[1] for r in rows])
    fig,ax=plt.subplots(figsize=(7,3.2),layout='constrained')
    ax.bar(np.arange(1,len(sizes)+1),sizes,color='#356ca5')
    ax.set(xlabel='Leiden community, ranked by size',ylabel='Number of tracks',
           title=f'k={k}: weighted Leiden community sizes')
    save_plot(fig,folder,'community_sizes')
    fig,ax=plt.subplots(figsize=(7,max(3.4,min(9,.28*len(rows)+1.4))))
    fig.subplots_adjust(left=.12,right=.97,bottom=.13,top=.78)
    show=min(len(rows),25)
    fraction=contingency[:show]/contingency[:show].sum(axis=1,keepdims=True)
    ax.barh(np.arange(show),fraction[:,0],color='#326eae',label='E5 domain 1')
    ax.barh(np.arange(show),fraction[:,1],left=fraction[:,0],
            color='#d97932',label='E5 domain 2')
    ax.set(yticks=np.arange(show),yticklabels=np.arange(1,show+1),
           xlabel='Fraction of Leiden community',ylabel='Leiden community',
           xlim=(0,1))
    ax.invert_yaxis()
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.02),ncol=2,frameon=False)
    save_plot(fig,folder,'e5_overlap')
    top_communities=min(len(rows),8)
    tag_total=Counter()
    for row in tag_rows:
        if row[0]==1:
            tag_total[row[1]]=row[5]
    tags=[tag for tag,_ in tag_total.most_common(10)]
    p={(r[0],r[1]):r[4] for r in tag_rows}
    values=np.array([[p[(c,tag)] for tag in tags]
                     for c in range(1,top_communities+1)],dtype=float)
    fig,ax=plt.subplots(figsize=(8,max(3,.4*top_communities+1)),layout='constrained')
    im=ax.imshow(values,aspect='auto',cmap='Blues',vmin=0)
    ax.set_xticks(np.arange(len(tags)),tags,rotation=35,ha='right')
    ax.set_yticks(np.arange(top_communities),
                  [f'C{c} (n={sizes[c-1]})' for c in range(1,top_communities+1)])
    ax.set_title(f'k={k}: genre prevalence in largest Leiden communities')
    fig.colorbar(im,ax=ax,label='Fraction among genre-labeled tracks')
    save_plot(fig,folder,'genre_profiles')
    with np.load(spectral,allow_pickle=False) as data:
        coords=data['embedding']
    assignment=np.load(folder/'membership.npy')
    shown=min(8,len(rows))
    fig=plt.figure(figsize=(9.6,4.25))
    axes=[fig.add_axes([.02,.08,.34,.84],projection='3d'),
          fig.add_axes([.43,.08,.34,.84],projection='3d')]
    palette=plt.get_cmap('tab10')
    for panel,(ax,azimuth) in enumerate(zip(axes,(-60,35)),1):
        other=assignment>shown
        if other.any():
            ax.scatter(*coords[other].T,s=3,color='.78',alpha=.32,
                       depthshade=False,label=f'Other (n={int(other.sum())})')
        for c in range(1,shown+1):
            mask=assignment==c
            ax.scatter(*coords[mask].T,s=4,alpha=.67,depthshade=False,
                       color=palette(c-1),label=f'C{c} (n={int(mask.sum())})')
        ax.set(xlabel=r'$q_2$',ylabel=r'$q_3$',
               zlabel='' if panel==1 else r'$q_4$')
        ax.xaxis.label.set_size(13)
        ax.yaxis.label.set_size(13)
        ax.zaxis.label.set_size(11)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.set_major_locator(MaxNLocator(3))
            axis.set_tick_params(labelsize=10,pad=0)
        ax.view_init(elev=23,azim=azimuth)
        ax.set_box_aspect((1,1,1))
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='center left',bbox_to_anchor=(.81,.50),
               fontsize=12,markerscale=2.5,frameon=False)
    save_plot(fig,folder,'e5_embedding_communities')


def run_one(k, nodes, tags):
    folder=HERE/f'k{k:03d}'
    folder.mkdir(exist_ok=True)
    a, upper, graph, graph_hash=load_graph(k)
    with np.load(E4_E5/f'k{k:03d}'/'spectral_data.npz',allow_pickle=False) as data:
        indices=data['global_node_index']; e5=data['domain_id']
    assert np.array_equal(indices,np.arange(2000)) and np.array_equal(np.unique(e5),[1,2])
    trials=[]; raw_memberships=[]
    for seed in SEEDS:
        part=la.find_partition(graph,la.ModularityVertexPartition,
                               weights='weight',n_iterations=-1,seed=seed)
        membership=np.asarray(part.membership,dtype=int)
        q=float(graph.modularity(membership.tolist(),weights='weight'))
        assert abs(q-float(part.quality()))<1e-8
        trials.append((seed,q,len(part),min(part.sizes()),max(part.sizes())))
        raw_memberships.append(membership)
    write_csv(folder/'restarts.csv',
              ['seed','weighted_modularity','community_count','min_size','max_size'],trials)
    best=max(range(len(SEEDS)),key=lambda i:(trials[i][1],-SEEDS[i]))
    assignment=canonicalize(raw_memberships[best])
    np.save(folder/'membership.npy',assignment)
    stability=[adjusted_rand_score(raw_memberships[i],raw_memberships[j])
               for i,j in combinations(range(len(SEEDS)),2)]
    stats,contingency,all_tags,tag_rows=community_statistics(a,upper,assignment,e5,
                                                             nodes,tags,k,folder)
    graph_cut=assignment[upper.row]!=assignment[upper.col]
    q=trials[best][1]
    overlap={
        'adjusted_rand_index':float(adjusted_rand_score(assignment,e5)),
        'normalized_mutual_info':float(normalized_mutual_info_score(assignment,e5)),
    }
    agreement=genre_agreement(upper,assignment,tags)
    metrics={
        'k':k,'nodes':a.shape[0],'edges':graph.ecount(),
        'graph_sha256':graph_hash,'objective':'weighted Newman-Girvan modularity',
        'resolution_gamma':1.0,'iterations':'until no quality increase',
        'seeds':list(SEEDS),'selected_seed':SEEDS[best],
        'weighted_modularity':q,'community_count':len(stats),
        'community_size_min':int(min(row[1] for row in stats)),
        'community_size_median':float(np.median([row[1] for row in stats])),
        'community_size_max':int(max(row[1] for row in stats)),
        'community_size_top_5':[int(row[1]) for row in stats[:5]],
        'all_communities_connected':True,
        'cross_community_edges':int(graph_cut.sum()),
        'cross_community_edge_fraction':float(graph_cut.mean()),
        'cross_community_weight_fraction':float(upper.data[graph_cut].sum()/upper.data.sum()),
        'restart_ari_min':float(min(stability)),
        'restart_ari_median':float(np.median(stability)),
        'e5_nodal_domain_comparison':overlap,
        'genre_edge_agreement':agreement,
    }
    write_json(folder/'metrics.json',metrics)
    write_csv(folder/'nodes.csv',
              ['global_node_index','track_id','artist_id','community_id','e5_domain_id',
               'genres','mood_theme'],
              ((i,n['track_id'],n['artist_id'],int(assignment[i]),int(e5[i]),
                '|'.join(n['genre']),'|'.join(n['mood/theme']))
               for i,n in enumerate(nodes)))
    extra_interpretation(assignment,nodes,folder)
    plot_results(k,folder,stats,contingency,tag_rows,
                 E4_E5/f'k{k:03d}'/'spectral_data.npz')
    print(f'k={k}: {len(stats)} communities, Q={q:.6f}, selected seed={SEEDS[best]}',flush=True)
    return metrics,assignment


def main():
    nodes=[json.loads(line) for line in (RUN/'nodes.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(nodes)==2000 and [n['node_index'] for n in nodes]==list(range(2000))
    tags=[set(n['genre']) for n in nodes]
    results={}
    for k in KS:
        results[k]=run_one(k,nodes,tags)
    a=results[3][1]; b=results[64][1]
    between={'adjusted_rand_index':float(adjusted_rand_score(a,b)),
             'normalized_mutual_info':float(normalized_mutual_info_score(a,b)),
             'same_community_id_fraction_not_meaningful':None}
    write_json(HERE/'cross_k_comparison.json',between)
    write_csv(HERE/'comparison.csv',
              ['k','nodes','edges','community_count','weighted_modularity',
               'size_min','size_median','size_max','cross_weight_fraction',
               'restart_ari_median','e5_ari','e5_nmi','selected_seed'],
              ((k,m['nodes'],m['edges'],m['community_count'],m['weighted_modularity'],
                m['community_size_min'],m['community_size_median'],m['community_size_max'],
                m['cross_community_weight_fraction'],m['restart_ari_median'],
                m['e5_nodal_domain_comparison']['adjusted_rand_index'],
                m['e5_nodal_domain_comparison']['normalized_mutual_info'],
                m['selected_seed']) for k,(m,_) in results.items()))
    write_json(HERE/'environment.json',
               {'python':sys.version.split()[0],'igraph':ig.__version__,
                'leidenalg':la.__version__,'numpy':np.__version__,
                'scipy':__import__('scipy').__version__,
                'sklearn':__import__('sklearn').__version__,
                'matplotlib':matplotlib.__version__})
    print('Cross-k ARI:',between['adjusted_rand_index'])


if __name__=='__main__':
    main()

import numpy as np
import pandas as pd
from grelu.sequence.format import convert_input_type
from grelu.interpret.motifs import scan_sequences
from grelu.transforms.prediction_transforms import Specificity, Aggregate
import torch
import os, sys
from captum.attr import InputXGradient, Saliency
from scipy.signal import find_peaks
from collections import defaultdict
from pymemesuite.common import Sequence, Background, Array, Alphabet, MotifFile
from pymemesuite.fimo import FIMO
 

src_dir = f'{os.path.dirname(__file__)}/../src/decima/'
sys.path.append(src_dir)
from read_hdf5 import extract_gene_data

def read_meme_file(file):
    motiffile = MotifFile(file)
    motifs = []
    while True:
        motif = motiffile.read()
        if motif is None:
            break
        motifs.append(motif)

    with open(file, 'r') as f:
        l = f.readlines()
    names = [x.split(' ')[1].split('.')[0] for x in l if x.startswith('MOTIF')]
    return motifs, names

def scan(seq, motifs, names, pthresh):
    seq = Sequence(seq, name=b'')
    fimo = FIMO(both_strands=True, threshold=pthresh)
    bg = Background(alphabet = Alphabet.dna(), frequencies = Array([1/4]*4))
    out = defaultdict(list)
    for motif, name in zip(motifs, names):
        match = fimo.score_motif(motif, [seq], bg).matched_elements
        for m in match:
            out["motif"].append(name)
            out["start"].append(m.start)
            out["end"].append(m.stop)
            out["strand"].append(m.strand)
            out["pval"].append(m.pvalue)
    
    return pd.DataFrame(out)

def attributions(gene, tasks, h5_file, model, device, off_tasks=None, transform="specificity", 
                method=InputXGradient, **kwargs):

        seq, mask = extract_gene_data(h5_file, gene, merge=False)
        inputs = torch.vstack([seq, mask])
        tss_pos = np.where(mask[0] == 1)[0][0]
        
        if transform == "specificity":
            model.add_transform(Specificity(on_tasks=tasks, off_tasks=off_tasks, model=model, compare_func='subtract'))
        elif transform == "aggregate":
            model.add_transform(Aggregate(tasks=tasks, task_aggfunc="mean", model=model))
    
        model = model.eval()
        device = torch.device(device)
    
        attributer = method(model.to(device))
        with torch.no_grad():
            attr = attributer.attribute(inputs.to(device),**kwargs).cpu().numpy()[:4]
    
        model.reset_transform()
        return seq, tss_pos, attr